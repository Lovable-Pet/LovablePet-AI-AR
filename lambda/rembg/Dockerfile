# Lambda image for background removal (handler.py). Build from the repo root:
#   docker build -f lambda/rembg/Dockerfile -t lovablepet-rembg .
# Test locally with the runtime emulator in the base image (the limits are the function's):
#   docker run -p 9000:8080 -e REMBG_KEY=test --memory=3008m --memory-swap=3008m --cpuset-cpus=0,1 lovablepet-rembg
#   REMBG_URL=http://localhost:9000/2015-03-31/functions/function/invocations REMBG_KEY=test ...

# the model rembg downloads for bria-rmbg, checked against rembg's own sha256,
# rewritten to fit in 3008 MB (slim_model.py). The rewrite's sha256 is the file
# whose masks were compared with the original's (same on Windows and Linux).
FROM public.ecr.aws/lambda/python:3.13 AS model
RUN pip install --no-cache-dir onnx==1.23.2 numpy==2.4.4
RUN python -c "import hashlib, urllib.request; \
p = '/tmp/bria-rmbg-2.0.onnx'; \
urllib.request.urlretrieve('https://github.com/danielgatis/rembg/releases/download/v0.0.0/bria-rmbg-2.0.onnx', p); \
h = hashlib.sha256(open(p, 'rb').read()).hexdigest(); \
assert h == '5b486f08200f513f460da46dd701db5fbb47d79b4be4b708a19444bcd4e79958', h"
COPY lambda/rembg/slim_model.py /tmp/
RUN python /tmp/slim_model.py /tmp/bria-rmbg-2.0.onnx /opt/bria-rmbg.onnx \
    && python -c "import hashlib; \
h = hashlib.sha256(open('/opt/bria-rmbg.onnx', 'rb').read()).hexdigest(); \
assert h == '2f6907526ffa9cdae0b0c814b881c6ad7d218baf5cbdb8966c870428df841c78', h" \
    && chmod 644 /opt/bria-rmbg.onnx

FROM public.ecr.aws/lambda/python:3.13

RUN pip install --no-cache-dir onnxruntime==1.30.0 numpy==2.4.4 pillow==12.3.0
COPY --from=model /opt/bria-rmbg.onnx /opt/bria-rmbg.onnx

# glibc keeps freed blocks below its mmap threshold, which grows to 32 MB as big
# blocks are freed: the pass then peaks at ~3.3 GB. With blocks over 1 MB always
# mmapped (returned to the system when freed) it peaks at ~1.9 GB.
ENV MALLOC_MMAP_THRESHOLD_=1048576 \
    MALLOC_TRIM_THRESHOLD_=1048576

COPY api/bria.py lambda/rembg/handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
