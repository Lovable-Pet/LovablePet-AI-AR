"""BRIA RMBG 2.0 rewritten to fit the 3008 MB Lambda: the same arithmetic in another order.

rembg's bria-rmbg-2.0.onnx peaks at ~7 GB on CPU, this one at ~1.9 GB (Lambda
image under a 3008 MB limit, 2026-10-30). Masks on the 29 eval/fixture photos
are within 1/255 of the original's. Three rewrites:

1. DeformConv. The file is opset 14, before ONNX had a deformable conv, so each
   of the 20 became ~60 nodes: 4 GatherND, one per bilinear corner, on a
   k*H x k*W im2col tensor. Each is now one DeformConv (opset 19). Of the ops in
   the graph only ReduceMean changed from 14 to 19 (axes attribute -> input).
2. Chunks. DeformConv's own im2col buffer is C*k*k*H*W floats, 784 MB for
   decoder_block1's 7x7: a conv whose buffer would pass COL_MB runs as
   input-channel chunks added one by one.
3. Tail. The graph ends with sigmoid(conv1x1(concat(resize(p1), conv3x3(h)))),
   a 768 MB resize and a 960 MB concat at 1024x1024. The 1x1 conv commutes with
   the bilinear resize and folds into the 3x3 conv before it (no activation in
   between), so the tail is resize(conv1x1(p1)) + conv3x3(h), one channel each.

usage: python slim_model.py SRC.onnx DST.onnx
"""

import collections
import re
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper

OPSET = 19
COL_MB = 128


def _arr(inits, name):
    return numpy_helper.to_array(inits[name])


def native_deform(g, inits, by_name):
    """{final conv's name: nodes replacing it} for every deformable conv."""
    scopes = collections.OrderedDict()
    for n in g.node:
        mt = re.match(r"(.*/atrous_conv)/", n.name)
        if mt:
            scopes.setdefault(mt.group(1), []).append(n)
    assert len(scopes) == 20, len(scopes)
    two = "slim/two"
    g.initializer.append(numpy_helper.from_array(np.array(2.0, dtype=np.float32), two))
    replace = {}
    for s, ns in scopes.items():
        off, mod, final = by_name[s + "/offset_conv/Conv"], by_name[s + "/modulator_conv/Conv"], by_name[s + "/Conv"]
        assert off.input[0] == mod.input[0], s
        sig = [n for n in ns if n.op_type == "Sigmoid"]
        assert len(sig) == 1 and sig[0].input[0] == mod.output[0], s
        # BiRefNet's mask is 2 * sigmoid(modulator_conv(x))
        assert any(n.op_type == "Mul" and any(i in inits and inits[i].dims == [] and float(_arr(inits, i)) == 2.0
                                              for i in n.input) for n in ns), s
        x, w, b = off.input[0], final.input[1], final.input[2]   # b: BatchNorm folded in by the exporter
        c_in, (kh, kw) = inits[w].dims[1], list(inits[w].dims[2:])
        attr = {a.name: a for a in off.attribute}
        assert list(attr["strides"].ints) == [1, 1], s
        # the decomposed body adds a constant sampling grid [1, 1, k*k, 2, H, W]: the output size
        h, wd = [list(inits[i].dims) for n in ns for i in n.input if i in inits and len(inits[i].dims) == 6][0][4:]
        chunks = 1
        while c_in * kh * kw * h * wd * 4 / chunks / 2**20 > COL_MB and c_in % (chunks * 2) == 0:
            chunks *= 2
        conv = dict(kernel_shape=[kh, kw], pads=list(attr["pads"].ints), strides=[1, 1], dilations=[1, 1],
                    group=1, offset_group=1)
        mask = s + "/mask"
        new = [helper.make_node("Mul", [sig[0].output[0], two], [mask], name=s + "/mask_scale")]
        if chunks == 1:
            new.append(helper.make_node("DeformConv", [x, w, off.output[0], b, mask], [final.output[0]],
                                        name=s + "/DeformConv", **conv))
        else:
            step, wa = c_in // chunks, _arr(inits, w)
            sizes = s + "/chunk_sizes"
            g.initializer.append(numpy_helper.from_array(np.full(chunks, step, dtype=np.int64), sizes))
            xs = [f"{s}/x_{i}" for i in range(chunks)]
            new.append(helper.make_node("Split", [x, sizes], xs, name=s + "/split", axis=1))
            acc = None
            for i in range(chunks):
                wi = f"{w}/chunk_{i}"
                g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(wa[:, i * step:(i + 1) * step]), wi))
                yi = f"{s}/y_{i}"
                new.append(helper.make_node("DeformConv", [xs[i], wi, off.output[0], b if i == 0 else "", mask], [yi],
                                            name=f"{s}/DeformConv_{i}", **conv))
                if acc is not None:
                    out = final.output[0] if i == chunks - 1 else f"{s}/acc_{i}"
                    new.append(helper.make_node("Add", [acc, yi], [out], name=f"{s}/acc_{i}"))
                    yi = out
                acc = yi
        replace[final.name] = new
    return replace


def fold_tail(g, inits, by_name):
    """{name: nodes replacing it} for the 1024x1024 tail (module docstring, 3)."""
    prod = {o: n for n in g.node for o in n.output}
    head = by_name["/model/decoder/conv_out1/conv_out1.0/Conv"]
    cat = prod[head.input[0]]
    resize, ipt = prod[cat.input[0]], prod[cat.input[1]]
    assert (cat.op_type, resize.op_type, ipt.op_type) == ("Concat", "Resize", "Conv")
    sizes = prod[resize.input[3]]                       # Concat([1, 192], [H, W])
    assert sizes.op_type == "Concat" and _arr(inits, sizes.input[0]).tolist() == [1, 192]

    w, b = _arr(inits, head.input[1]), _arr(inits, head.input[2])       # [1, 240, 1, 1], [1]
    w_p1, w_ipt_out = w[:, :192], w[0, 192:, 0, 0].astype(np.float64)   # [1, 192, 1, 1], [48]
    w_ipt, b_ipt = _arr(inits, ipt.input[1]).astype(np.float64), _arr(inits, ipt.input[2]).astype(np.float64)
    w_fold = (w_ipt_out[:, None, None, None] * w_ipt).sum(axis=0)[None].astype(np.float32)   # [1, 64, 3, 3]
    b_fold = np.array([(w_ipt_out * b_ipt).sum()], dtype=np.float32)

    p = "slim/tail"
    for name, value in [("w_p1", w_p1), ("b_p1", b), ("w_ipt", w_fold), ("b_ipt", b_fold),
                        ("one_one", np.array([1, 1], dtype=np.int64))]:
        g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(value), f"{p}/{name}"))
    attrs = lambda n: {a.name: helper.get_attribute_value(a) for a in n.attribute}
    return {head.name: [
        helper.make_node("Conv", [resize.input[0], p + "/w_p1", p + "/b_p1"], [p + "/p1"], name=p + "/conv_p1",
                         **attrs(head)),
        helper.make_node("Concat", [p + "/one_one", sizes.input[1]], [p + "/sizes"], name=p + "/sizes", axis=0),
        helper.make_node("Resize", [p + "/p1", "", "", p + "/sizes"], [p + "/p1_up"], name=p + "/resize",
                         **attrs(resize)),
        helper.make_node("Conv", [ipt.input[0], p + "/w_ipt", p + "/b_ipt"], [p + "/ipt"], name=p + "/conv_ipt",
                         **attrs(ipt)),
        helper.make_node("Add", [p + "/p1_up", p + "/ipt"], [head.output[0]], name=p + "/add"),
    ]}


def main(src, dst):
    m = onnx.load(src)
    assert [(o.domain, o.version) for o in m.opset_import] == [("", 14)]
    g = m.graph
    inits = {t.name: t for t in g.initializer}
    by_name = {n.name: n for n in g.node}
    replace = native_deform(g, inits, by_name) | fold_tail(g, inits, by_name)

    nodes = [r for n in g.node for r in replace.get(n.name, [n])]
    # keep what still feeds the output: drops the decomposed bodies and the old tail
    needed = {o.name for o in g.output}
    keep = []
    for n in reversed(nodes):
        if any(o in needed for o in n.output):
            keep.append(n)
            needed.update(i for i in n.input if i)
    keep.reverse()
    for n in keep:
        axes = [a for a in n.attribute if a.name == "axes"]
        if n.op_type == "ReduceMean" and axes:          # ReduceMean-18: axes is an input
            name = n.name + "/axes"
            g.initializer.append(numpy_helper.from_array(np.array(axes[0].ints, dtype=np.int64), name))
            n.input.append(name)
            needed.add(name)
            n.attribute.remove(axes[0])
    del g.node[:]
    g.node.extend(keep)
    # in place: copying the kept weights into a new list would hold the model twice
    for i in reversed(range(len(g.initializer))):
        if g.initializer[i].name not in needed:
            del g.initializer[i]
    # the exporter's shape annotations name tensors that are gone; onnxruntime infers its own
    del g.value_info[:]
    del m.opset_import[:]
    m.opset_import.extend([helper.make_opsetid("", OPSET)])
    ops = collections.Counter(n.op_type for n in g.node)
    onnx.save(m, dst)
    del m, g, inits, by_name, replace, nodes, keep
    onnx.checker.check_model(dst)
    print(f"{dst}: {sum(ops.values())} nodes, DeformConv {ops['DeformConv']}, GatherND {ops['GatherND']}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
