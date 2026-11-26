"""What goes with the model that reads the photographs.

The vision calls (photo gate, outline, layout, face, body) were written and
their prompts tuned on gpt-4o. Another model reads the same photographs but
takes the same sentences differently, so what a model needs besides its
name is kept here, in one place, and follows the model: setting
OPENAI_MODEL is the whole switch.

gpt-6-luna (2026-11-13, tools/cost/fair_compare.py): about a tenth of
gpt-4o's price for the reading ($0.0056 with the extra requests and
views below, measured on 60 builds, against $0.053 a dog). On 19 new
shelter photos judged blind the two could not be told apart (coat right 16
and 15, template 15 and 16). It thinks before answering unless told not to,
and bills the thinking as output (reasoning_effort). It follows the layout
question's sentences to the letter where gpt-4o reads past them
(pattern_prompt, api/pattern_classify.py).
"""

from __future__ import annotations

from typing import Any

# The reading model when OPENAI_MODEL is not set. gpt-4o until 2026-11-13;
# OPENAI_MODEL=gpt-4o in the server's .env is the way back (then
# `docker compose up -d`), and brings gpt-4o's wording with it.
DEFAULT_MODEL = "gpt-6-luna"

PROFILES: dict[str, dict[str, Any]] = {
    # gpt-4o takes no reasoning_effort (the API refuses the parameter)
    "gpt-4o": {"pattern_prompt": "tuned"},
    # layout_requests, face_requests, body_requests: the question in this
    # many separate requests, each of the usual samples, voted together.
    # gpt-6-luna's samples of one request agree with each other (90%) more
    # than two asks an hour apart do (79%), so one request's vote is nearly
    # one opinion; three requests cost $0.001.
    # view_margins, face_pads: what each of those requests looks at. The
    # same picture sent three times at once comes back the same (88%, and a
    # tan dog's chest unknown 27 of 27 where the build an hour before said
    # cream 3 of 3), so the requests would add votes and no opinions. Each
    # gets its own view instead: the photograph with that much grey round
    # it (sampling.with_margin; the layout and body calls), the head
    # cropped at that padding beside the usual 0.10 (face_crop.crop_views).
    # Between such views two samples agree 80%, as between two asks an hour
    # apart, and the layout call scored the same against the 58 keyed dogs
    # (53 right; 52 with three identical requests) - tools/cost/asks_replay.py,
    # 2026-11-14.
    # rules: what the rules after the calls do differently (api/lite.py).
    # Several of those rules were written against gpt-4o's habits and undo
    # an answer gpt-6-luna gets right:
    #   piebald_by_head   a piebald whose crown the face call reads light is
    #                     kept when the head crop measures patched
    #   patches_by_body   body patches are drawn when every sample of the
    #                     body call saw them, whatever the gate's view word
    #   dark_muzzle_by_asks  a warm coat's dark muzzle from a third of the
    #                     votes of face_requests requests, not all of one
    #   body_by_asks      the chest, belly and feet from the samples of all
    #                     body_requests requests, not the first's alone
    #   plain_front       a "dark" chest or belly is not painted on a coat
    #                     with no dark colour in it (shadow)
    #   irish_as_seen     an irish coat's legs and chest white only where the
    #                     calls saw white: toes, a chest patch
    #   white_trunk       irish read dark over white with a white trunk
    #                     (body_extra's question) is a white dog with a dark head
    #   pale_warm_coat    a coat read white or cream with a tan crown that
    #                     measures between the two is fawn all over
    #   warm_coat_measured  a cream coat with a tan crown becomes tan all
    #                     over on the measurement only, not on the crown's votes
    #   black_brindle_plain  a brindle on a coat that stays black is drawn as
    #                     the black coat (the overlay's tan made a black dog brown)
    #   plain_brows       eyebrow dots paler than the head are not drawn, off
    #                     a point pattern (two pale discs over the eyes)
    #   plain_crown       a solid white or cream coat gets no warm cap from the
    #                     crown answer alone
    #   front_not_deeper  a solid white or cream coat's chest and belly are
    #                     not painted tan
    #   chihuahua_by_all  the template: a chihuahua by some of the shape call's
    #                     samples only is the looks' model (template_pick.combine)
    #   maltese_by_looks  the template: a toy poodle the shape call named, all
    #                     samples or not, is the maltese the looks are sure of
    #                     (template_pick.combine)
    #   mask_seen_light   a dark muzzle whose own region measures light is not drawn
    #   faded_black_all_over  a black coat goes charcoal only when every measured
    #                     one of skull, back and legs is charcoal
    #   coloured_trunk    a piebald whose trunk is coloured with no patches on
    #                     the body is an irish coat (the white_trunk reversed)
    #   grey_any_border   a warm coat with hardly any warm fur and a grey whole
    #                     coat is drawn dull off any border (lite.coat_without_warm_fur)
    #   tall_box_by_call  off a head box much taller than wide (the back in the
    #                     crop), the face samples win where the crop says the
    #                     opposite: no full blaze, no grey or dark cheeks
    #   black_back_kept   a coat read black that the crown answer "gray" turned
    #                     grey is black again when its back measures dark
    #                     (lite.black_back_kept)
    #   plain_on_close_vote  a marked layout that beat solid by a bare majority
    #                     is built solid (pattern_classify.classify_pattern)
    #   piebald_by_dark_eyes  a piebald with a light crown and too little colour
    #                     in the head crop is kept when every face sample saw
    #                     dark fur round the eyes and the patches are dark
    #   sable_tips_black  a sable's second colour named gray (the question's
    #                     "grey-black overlay") is black (pattern_classify)
    #   sable_by_most     a sable is drawn when two layout samples in three
    #                     say so (of nine), the back question as before
    #                     (lite.sable_doubt)
    #   sable_front_tipped  a sable's dark chest and belly are the overlay's
    #                     tipping over the base, not regions in the second colour
    #   brindle_stripes_black  a brindle's stripes named gray are black
    #   brindle_front_striped  a brindle's dark chest and belly are left to the
    #                     stripes, not painted as regions in the stripe colour
    #   differences_caveat  the fit verdict: the parts named as differing from
    #                     the model are a caveat however many, never "poor" by
    #                     their count (body_shape.match_from)
    # body_extra: questions the body call asks this model besides its own
    # (body_colours.EXTRA_QUESTIONS).
    "gpt-6-luna": {"reasoning_effort": "none", "pattern_prompt": "literal",
                   "layout_requests": 3, "face_requests": 3, "body_requests": 3,
                   "view_margins": (0.0, 0.06, 0.12), "face_pads": (0.0, 0.2),
                   "body_extra": ("trunk",),
                   "rules": ("piebald_by_head", "patches_by_body", "dark_muzzle_by_asks", "body_by_asks",
                             "plain_front", "irish_as_seen", "white_trunk", "coloured_trunk", "pale_warm_coat",
                             "warm_coat_measured", "black_brindle_plain",
                             "plain_brows", "plain_crown", "front_not_deeper", "chihuahua_by_all",
                             "maltese_by_looks",
                             "mask_seen_light", "faded_black_all_over", "black_back_kept", "plain_on_close_vote",
                             "sable_tips_black", "sable_by_most", "sable_front_tipped",
                             "brindle_stripes_black", "brindle_front_striped",
                             "piebald_by_dark_eyes", "tall_box_by_call", "grey_any_border",
                             "differences_caveat")},
}


def profile(model: str | None) -> dict[str, Any]:
    """The model's entry; a model without one is asked as gpt-4o is."""
    return PROFILES.get(model or "", {})


def has_rule(model: str | None, name: str) -> bool:
    return name in profile(model).get("rules", ())


__all__ = ["DEFAULT_MODEL", "PROFILES", "has_rule", "profile"]
