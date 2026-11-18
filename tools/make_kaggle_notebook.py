#!/usr/bin/env python3
"""Generate standalone 24-image (8x3) Kaggle TPU v5e-8 production benchmark notebooks."""
import json
import os

PROD_PROMPTS = [
    {
        "id": 1,
        "name": "sky01_sky_view",
        "video": "sky-01-blue-sky",
        "scene": "sky_view",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a wide daytime sky arching over a flat horizon. The paper is flat and even to every edge and corner. In the upper-left corner of the frame, a small hand-drawn sun shines as a simple open circle with six short, delicate radiating lines. Across the upper half of the composition, a few gentle horizontal strokes of pale cerulean blue indicate the calm dome of the clear atmosphere. In the centre, the vast open expanse remains clean and empty, emphasising the quiet stillness of the open air. Across the lower third of the frame, a single straight horizontal pencil line demarcates the distant ground, with two tiny minimalist hills sketched on the far right. On the left side of the lower area, three tiny stick-like grass tufts stand quietly along the ground line. The lines are clean dark grey pencil strokes with a soft wash of sky-blue watercolour across the upper expanse, without heavy shading or paper grain. The overall composition is balanced, airy, and simple, with generous empty beige paper conveying infinite space.",
        "seed": 6001,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 2,
        "name": "sky01_early_beliefs",
        "video": "sky-01-blue-sky",
        "scene": "early_beliefs",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of an antique observation of dust motes dancing in a single shaft of light. The paper is flat and even to every edge and corner. In the upper-left corner of the frame, two straight parallel diagonal lines trace a bright narrow beam entering an open window frame. In the centre of the frame, inside this diagonal light corridor, seven tiny circular dust specks float serenely at varying heights. On the right side of the frame, a simple robed figure in profile stands looking toward the shaft, sketched with minimal contour lines and an outstretched hand pointing toward the floating particles. Across the lower third, a plain flat wooden floor line grounds the room, leaving the surrounding wall space clear and unadorned. In the upper-right corner, empty blank paper preserves the uncluttered feeling of the chamber. The lines are clean dark grey pencil outlines with a subtle warm yellow tint highlighting the beam and its tiny suspended dots, without shading or hatching. The overall composition is calm, thoughtful, and uncluttered, with ample negative space around the illuminated dust motes.",
        "seed": 6002,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 3,
        "name": "sky01_tyndall_experiment",
        "video": "sky-01-blue-sky",
        "scene": "tyndall_experiment",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of John Tyndall's historical laboratory experiment demonstrating light scattering in a glass tube. The paper is flat and even to every edge and corner. In the centre of the frame, a long horizontal cylindrical glass tube rests securely on two small wooden triangular stands. On the left side of the frame, a simple lamp housing projects a straight horizontal beam of white light directly into the left aperture of the tube. Inside the central section of the tube, tiny faint circular particles scatter short radiating lines of vivid blue light upward and outward toward the viewer. On the right side of the frame, the transmitted beam exits the glass tube as a diminished warm reddish ray hitting a plain square screen. Across the lower third, a straight horizontal workbench line supports the entire apparatus, leaving the upper background completely blank. The lines are clean dark grey pencil contours with bright blue highlights inside the cylinder and a faint amber accent on the exit beam, without crosshatching or texture. The overall composition is orderly, scientific, and clear, with abundant bare paper surrounding the apparatus.",
        "seed": 6003,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 4,
        "name": "sky02_sunset_view",
        "video": "sky-02-red-sunsets",
        "scene": "sunset_view",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a setting sun touching an open sea horizon. The paper is flat and even to every edge and corner. In the centre of the frame, a low semicircle represents the glowing sun resting directly upon a straight horizontal line. Across the upper half of the composition, five long parallel horizontal pencil strokes washed in warm amber and crimson tint the evening sky. In the upper-left corner of the frame, three tiny silhouetted seagulls fly toward the glowing horizon with simple angular wings. On the right side of the frame, the sky transitions into a faint dusty lavender wash near the upper border, leaving wide areas of bare paper. On the left side of the frame, a soft radiant pencil halo arches gently above the descending solar disc. Across the lower third of the frame, gentle horizontal pencil ripples indicate the calm water surface reflecting the warm sunset glow in broken strokes. In the bottom corners, clear bare paper anchors the marine expanse. The lines are clean dark grey pencil outlines with delicate watercolour washes of orange, red, and dusky violet, without heavy shading or dark fills. The overall composition is serene, spacious, and balanced, surrounded by generous margins of empty beige paper.",
        "seed": 6004,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 5,
        "name": "sky02_blue_sky_recap",
        "video": "sky-02-red-sunsets",
        "scene": "blue_sky_recap",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of daytime blue skylight scattering in the atmosphere. The paper is flat and even to every edge and corner. In the upper-left corner of the frame, a high round sun emits a straight pencil beam downward into the atmospheric layer. In the centre of the frame, three diatomic molecules float as paired small circles touching each other. Around each molecule, fine radiating dashed blue lines scatter outward in multiple directions, illustrating the scattering of short blue wavelengths. On the right side of the frame, two straight continuous pencil lines tinted in pale yellow pass directly through the open space without deviation. Across the lower third of the frame, a single level horizontal baseline indicates the ground, where a tiny minimalist tree stands on the far right. Across the upper edge, open blank beige paper preserves a light and airy atmosphere. The lines are clean dark grey pencil contours with bright cerulean blue accents along the scattered paths and a gentle gold touch on the main beam, without shading or crosshatching. The overall composition is orderly, diagrammatic, and clear, with abundant negative space around the floating molecules.",
        "seed": 6005,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 6,
        "name": "sky02_ancient_beliefs",
        "video": "sky-02-red-sunsets",
        "scene": "ancient_beliefs",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of an ancient natural philosopher observing the sunset from a stone terrace. The paper is flat and even to every edge and corner. On the left side of the frame, a robed scholar stands in profile, holding a simple sighting rod toward the distant horizon. In the centre of the frame, wide empty space conveys the quiet evening air between the scholar and the setting sun. On the right side of the frame, a low circular sun rests upon a distant mountain ridge, drawn with four small flickering flame shapes emerging from its upper edge. Across the lower third of the frame, a simple horizontal stone parapet line grounds the terrace, with three small masonry joints sketched in clean strokes. In the upper half of the composition, vast unadorned beige paper evokes an ancient twilight sky. The lines are clean dark grey pencil outlines with a single wash of muted red ochre illuminating the fiery sun disc, without shading or tone. The overall composition is contemplative, historical, and restrained, surrounded by generous margins of warm paper.",
        "seed": 6006,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 7,
        "name": "sky04_hook",
        "video": "sky-04-rainbows",
        "scene": "hook",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a complete circular rainbow floating above a flat landscape. In the centre of the frame, a complete circular ring is drawn with three thin concentric pencil lines, tinted gently with faint red on the outer ring and faint blue on the inner ring. In the upper-left corner, a simple tiny outline of an aeroplane is drawn with five simple grey strokes, flying horizontally toward the right. Across the lower third, a faint horizontal line indicates the distant flat ground, drawn with one soft pencil stroke. On the right side of the frame, empty beige paper provides calm negative space around the circular arc. The lines are clean dark grey pencil lines; only small red, orange or blue touches on the key idea; no shading, no texture. The overall composition is balanced, calm, and simple, with generous blank beige space.",
        "seed": 6007,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 8,
        "name": "sky04_ancient_view",
        "video": "sky-04-rainbows",
        "scene": "ancient_view",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of an ancient observer looking up at an arched sky band. On the left side of the frame, a single stick figure stands on a flat ground line, with head tilted upward toward the upper right. In the centre of the frame, a sweeping curved arch made of three light pencil arcs extends from the upper centre down toward the right. On the right side of the frame, two small puffy cloud outlines hover in the open air, drawn with simple wavy pencil strokes. Across the lower third, a plain horizontal ground line spans the frame, leaving the bottom area empty. The lines are clean dark grey pencil lines; only small red, orange or blue touches on the key idea; no shading, no texture. The overall composition is balanced, calm, and simple, with generous blank beige space.",
        "seed": 6008,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 9,
        "name": "sky04_early_experiment",
        "video": "sky-04-rainbows",
        "scene": "early_experiment",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a spherical glass flask filled with clear water resting on a wooden stand. In the centre of the frame, a large round circle represents the glass flask, drawn with two delicate concentric pencil outlines. On the left side of the frame, a straight horizontal beam enters the sphere near its upper edge, indicated by a single thin pencil line. Inside the sphere, the path angles downward toward the back wall with a light pencil stroke. Across the lower third, a simple horizontal tabletop line anchors the flask on a small wooden tripod stand. On the right side of the frame, open space keeps the focus directly on the glass vessel. The lines are clean dark grey pencil lines; only small red, orange or blue touches on the key idea; no shading, no texture. The overall composition is balanced, calm, and simple, with generous blank beige space.",
        "seed": 6009,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 10,
        "name": "sky05_rainbow_link",
        "video": "sky-05-auroras",
        "scene": "rainbow_link",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a full circular rainbow viewed from high in the sky. In the centre of the frame, a complete circle of faint concentric rings floats peacefully above a soft bank of low clouds. On the left side of the frame, several tiny curved pencil strokes indicate wispy cloud tops drifting horizontally across the paper. On the right side of the frame, a small outline of an airplane wing points inward toward the centre, drawn with four delicate grey lines. In the upper half of the paper, open empty space suggests a clear atmosphere with plenty of untouched beige background. Across the lower third of the scene, gentle undulating contours trace the fluffy upper boundary of the cloud blanket below. The lines are clean dark grey pencil lines; only small red, yellow and blue touches on the key idea; no shading, no texture. The coloured touches follow the circular outline in three thin adjacent rings, leaving the centre of the circle empty. The overall composition is balanced, calm and simple.",
        "seed": 6010,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 11,
        "name": "sky05_ancient_myths",
        "video": "sky-05-auroras",
        "scene": "ancient_myths",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of ancient mythological warriors looking up at glowing curtains in the sky. In the centre of the frame, two simple stick figures of warriors stand side by side on a slight hill crest, holding two round shields outlined with single circular pencil strokes. On the left side of the frame, three tall spears tilt gently toward the sky, drawn as straight grey vertical lines. On the right side of the frame, a lone pine tree is rendered with five zigzag pencil marks on a short bare trunk. Across the upper half of the paper, three soft sweeping ribbon curves arc from left to right, representing the mysterious night sky glow. The lines are clean dark grey pencil lines; only small green and red touches on the key idea; no shading, no texture. Subtle pale green and faint crimson tints brush lightly along the two upper wavy ribbon arcs, highlighting the celestial glow against the plain paper. The overall composition is balanced, calm and simple.",
        "seed": 6011,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 12,
        "name": "sky05_birkeland_origin",
        "video": "sky-05-auroras",
        "scene": "birkeland_origin",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of Kristian Birkeland's Terrella laboratory experiment. In the centre of the frame, a spherical metal globe suspended inside a transparent rectangular glass chamber stands on a small pedestal. On the left side of the frame, a cathode emitter is sketched as a cylindrical rod pointing directly toward the metal sphere. On the right side of the frame, several thin electrical wires curl loosely away from the glass chamber toward a simple vacuum pump outline. In the upper corners of the paper, open empty space highlights the clean scientific apparatus. The lines are clean dark grey pencil lines; only small blue and yellow touches on the key idea; no shading, no texture. Faint yellow touches illuminate the two magnetic poles of the suspended globe where luminous rings appear, while soft blue touches outline the glass walls of the vacuum box. The overall composition is balanced, calm and simple.",
        "seed": 6012,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 13,
        "name": "sauropod_pillar_legs",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "pillar_legs",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of the column-like leg bones of a giant sauropod compared with the bent leg bones of a mammal. On the left side of the frame, four straight vertical limbs resemble classical architectural columns, drawn with thick vertical pencil strokes showing stacked limb bones aligned straight beneath a heavy hip outline. In the centre, open empty beige paper separates the two anatomical comparisons. On the right side of the frame, four bent mammalian limbs are drawn with angled joints at the knee and ankle, showing zigzag load paths under a smaller pelvis. Across the upper third, thin curved pencil outlines indicate the contrasting back contours of both animals. The lines are clean dark grey pencil lines with small warm orange touches highlighting the straight vertical joint lines and no shading or background textures. In the bottom corners, tiny horizontal dashes indicate a flat ground plane beneath the straight feet. The overall composition is balanced, calm, and uncluttered, leaving generous empty paper across the edges.",
        "seed": 6013,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 14,
        "name": "sauropod_no_mastication",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "no_mastication",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a sauropod snout stripping foliage from a tree branch without chewing. On the left side of the frame, the slender profile of an elongated dinosaur jaw opens slightly toward a leafy branch, drawn with smooth pencil contours showing an arched snout and nostril opening. In the centre, five thin cylindrical peg-shaped teeth interlock like a small rake along the front of the jaw, grasping a slender curved conifer stem. On the right side of the frame, seven needle-covered branchlets spread outward into empty space, with several detached green leaf needles drifting downward toward the open mouth. Across the lower third, the smooth chin tapers into a long slender throat outline without any thick jaw muscles or bulging cheeks. The lines are clean dark grey pencil lines with small green and orange touches on the stripped foliage and no shading or textures. The overall composition is simple, balanced, and airy, leaving abundant bare beige paper surrounding the quiet interaction.",
        "seed": 6014,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 15,
        "name": "sauropod_chewing_cost",
        "video": "why-did-sauropod-dinosaurs-grow-so-huge",
        "scene": "chewing_cost",
        "prompt": "The image is a minimalist hand-drawn sketch on warm beige paper (#F5EBD7) of a mounted sauropod skeleton rearing toward the ceiling of an exhibition hall. In the centre of the frame, a towering sauropod skeleton rises vertically with its forelimbs much taller than its hind limbs, drawn with precise pencil outlines showing the ribcage, shoulder blades, and an upward-curving neck. On the left side of the frame, the colossal front limbs stand planted on the museum floor like massive upright pillars, surrounded by empty space. On the right side of the frame, the long tail slopes gently downward to balance the elevated neck, extending toward the right edge. Across the upper third, the small delicate skull tilts forward at the apex of the arched neck near the top margin. Across the lower third, a clean horizontal floor line anchors the upright legs with faint reflection marks. The lines are crisp dark grey pencil lines with subtle terracotta touches along the spine and no shading or background clutter. The overall composition is monumental, balanced, and serene, surrounded by generous margins of blank warm paper.",
        "seed": 6015,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 16,
        "name": "iceage_glaciers",
        "video": "why-do-human-survive-through-ice-age",
        "scene": "glaciers",
        "prompt": "The image is a wide cinematic realistic photograph of a vast Pleistocene mammoth steppe stretching toward towering glacial ice walls under a pale wintry twilight, captured from a low vantage point over frozen tundra. In the upper-left corner, a low winter sun hangs close to the jagged horizon, casting faint golden rays across thin wisps of frost mist. In the upper-right corner, the sky deepens into a frosty indigo gradient with no clouds. In the centre of the frame, a small herd of three woolly mammoths walks slowly across an expanse of dry blonde steppe grasses, their long curved tusks and thick shaggy brown coats dusted with white snow. On the left side of the frame, a solitary wind-sculpted limestone boulder stands embedded in the permafrost, draped in patches of dark green lichen. On the right side of the frame, jagged blue-white glacier cliffs rise dramatically against the distant mountain ridge. Across the lower third, the foreground consists of shadowed frozen earth, dark grey gravel and scattered patches of compacted snow that remain calm and dark. The lighting is low horizontal golden backlight from the setting winter sun, casting long deep blue shadows across the plains and giving the mammoth fur crisp amber highlights. The overall composition is expansive and desolate, with a stark natural palette of icy blue, dry amber grass, rich brown and dark charcoal earth that conveys the immense scale of the Ice Age wilderness.",
        "seed": 6016,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 17,
        "name": "iceage_clothing",
        "video": "why-do-human-survive-through-ice-age",
        "scene": "clothing",
        "prompt": "The image is a wide realistic photograph of Paleolithic sewing tools and tailored clothing materials arranged neatly on a weathered slab of dark grey slate inside an archaeological workshop setting. In the upper-left corner, a roll of thick reindeer hide shows its coarse greyish-tan fur, tightly bundled with twisted sinew cord. In the centre of the frame, two polished bone eyed needles rest side by side atop a folded piece of supple brown leather, each needle carved from ivory with a neat round eye hole threaded with pale plant fibre. On the right side of the frame, three pointed flint awls and two scraper tools made of dark brown chert lie alongside small bone beads dyed with red ochre powder. Across the lower third, the textured slate surface displays fine scratch marks and soft shadows beneath the tools, leading to a smooth dark border. In the upper-right background, the out-of-focus background falls off into deep warm umber tones without distraction. The lighting is soft diffused daylight from an overhead window to the left, casting gentle natural shadows and picking out the polished gleam on the ivory bone needles and the supple grain of the stitched leather seams. The overall composition is intimate and focused, with an earthy palette of warm umber, weathered ivory, red ochre and dark slate grey that highlights ancient technological craftsmanship.",
        "seed": 6017,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 18,
        "name": "iceage_shelter",
        "video": "why-do-human-survive-through-ice-age",
        "scene": "shelter",
        "prompt": "The image is a wide realistic photograph of a full-scale reconstructed Upper Paleolithic mammoth-bone hut standing on a snowy hillside at cold dawn. In the upper-left corner, the pale blue morning sky holds a thin sliver of the waning moon amidst faint pink sunrise clouds. In the centre of the frame, the dome-shaped dwelling stands five metres wide, its circular foundation built from thirty interlocking mammoth jawbones and large pelvis bones, topped with an intricate framework of interlocking curved ivory tusks and covered with heavy sewn animal hides anchored by river stones. On the right side of the frame, a thin plume of pale grey woodsmoke rises from a smoke hole at the top of the hide roof, drifting gently into the crisp air. On the left side of the frame, a drying rack made of tied birch poles holds strips of dried meat and two folded mammoth pelts. Across the lower third, windblown powdery snow covers the frozen soil, showing trampled paths around the entrance and small patches of pale yellow dead grass. The lighting is low angled morning sun from the right, casting long cold blue shadows across the snow and illuminating the ivory tusks with a warm golden sheen. The overall composition is balanced and atmospheric, with a crisp contrast between cool arctic blues, brilliant white snow and the warm golden-brown textures of weathered bone and fur.",
        "seed": 6018,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 19,
        "name": "stars_eye_tracking",
        "video": "why-do-stars-twinkle-use-frog-model-and",
        "scene": "eye_tracking",
        "prompt": "The image is a wide realistic photograph of one bright white star shining low above a dark rural hillside on a clear night. In the centre of the frame, the star glows with a soft halo and faint spikes. In the upper-left corner, dozens of fainter stars are scattered across a deep navy sky. Across the lower third, the black silhouette of rolling hills and three bare trees runs from edge to edge. On the right side of the frame, a thin band of pale blue twilight lingers above the hills. The lighting is natural starlight only, with no moon, so the land is almost black. The overall composition is calm and quiet, a single bright point against a dark, still landscape.",
        "seed": 6019,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 20,
        "name": "stars_space_vacuum",
        "video": "why-do-stars-twinkle-use-frog-model-and",
        "scene": "space_vacuum",
        "prompt": "The image is a wide realistic photograph taken from orbit of sharp white stars above the curved edge of Earth at night. Across the upper half of the frame, hundreds of crisp pinpoint stars sit steady in pure black space. Across the lower third, the dark night side of Earth curves gently, dotted with faint orange city lights. In the centre of the frame, a thin glowing blue line of atmosphere traces the horizon. On the left side of the frame, a pale green band of aurora shimmers above the planet. The lighting is faint starlight and city glow, with deep black shadows. The overall composition is clean and still, with tiny unblinking stars above a thin shell of air.",
        "seed": 6020,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 21,
        "name": "stars_fullscreen_space",
        "video": "why-do-stars-twinkle-use-frog-model-and",
        "scene": "fullscreen_space",
        "prompt": "The image is a wide cinematic realistic photograph of countless sharp, unwavering stars burning brightly in the deep black vacuum of space, viewed from high above Earth. In the upper-left corner, a dense cluster of brilliant white and pale blue pinpoint stars shines with needle-like clarity against absolute darkness. In the centre of the frame, several prominent stars glow with distinct pale gold and icy diamond hues, completely steady without any blurring or atmospheric haze. On the right side of the frame, the soft, curved edge of Earth's airless upper horizon clips the corner, showing a pitch-black rim. Across the lower third, deep outer space remains calm and empty, creating an unobstructed dark expanse beneath the stellar field. The lighting is pure starlight cutting through vacuum, casting crisp specular points without any glow, scatter, or chromatic halo. The overall composition is balanced and quiet, with a striking palette of jet black, cold silver, and gentle golden pinpoints that captures the stillness of cosmic space.",
        "seed": 6021,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 22,
        "name": "vikings_bone_scan",
        "video": "why-vikings-are-strong",
        "scene": "bone_scan",
        "prompt": "The image is a wide photorealistic 3D-rendered scientific visualisation of a cross-section through a human humerus bone, showing thick cortical bone density against a dark slate-grey background. In the upper-left corner, a soft pale blue volumetric light beam shines diagonally downward through the dark space. In the centre of the frame, a cut cylindrical segment of arm bone rests upright, displaying a dense creamy-white outer cortical wall and a porous honeycomb marrow lattice in pale ochre at its core. On the right side of the frame, fine microscopic bone structures are visible as translucent glowing rings called osteons, showing concentric lamellae in subtle silver and white tones. Across the lower third, the polished dark stone pedestal reflects the bone segment with a faint muted reflection, surrounded by deep shadows in the bottom corners. The lighting is focused studio illumination from the top left, casting precise shadows to the right and highlighting the structural texture of the mineral matrix. The overall composition is clean and analytical, with a cool high-contrast palette of ivory, slate, and charcoal that feels like an advanced biomechanics research laboratory display.",
        "seed": 6022,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 23,
        "name": "vikings_scandinavian_forest",
        "video": "why-vikings-are-strong",
        "scene": "scandinavian_forest",
        "prompt": "The image is a wide cinematic realistic photograph of an ancient Scandinavian pine forest at misty dawn, filled with mossy boulders and towering Scots pine trees. In the upper-left corner, thin morning mist drifts between tall straight tree trunks, tinted with pale amber light from the rising sun. In the centre of the frame, several massive granite boulders covered in thick dark emerald moss rest among ferns and fallen pine needles. On the right side of the frame, two mature Scots pines rise out of frame, their reddish-brown bark showing deep rough furrows and patches of grey lichen. Across the lower third, damp dark earth, damp peat soil, and black granite stones lie in deep calm shadow, creating an uncluttered dark foreground. In the bottom corners, fallen decayed wood and dark green moss dissolve into soft shadows with no sharp highlights. The lighting is soft diffused morning sunlight filtering through mist from the background, creating gentle glowing rim highlights on the trunks and deep cool shadows forward. The overall composition is spacious, quiet, and grounded, with an earthy palette of olive green, granite grey, amber, and deep charcoal that evokes the primeval Scandinavian boreal wilderness.",
        "seed": 6023,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
    {
        "id": 24,
        "name": "vikings_adze_carpentry",
        "video": "why-vikings-are-strong",
        "scene": "adze_carpentry",
        "prompt": "The image is a wide realistic photograph of an authentic Viking iron hand adze and a freshly hewn oak beam resting on a rustic workbench inside a timber workshop. In the upper-left corner, raw timber shavings of pale blond oak lie curled beside an unfinished timber joint. In the centre of the frame, a hand-forged iron adze with a dark curved socket and sharp cutting edge is lodged lightly into the surface of a squared dark brown oak timber. On the right side of the frame, two wooden mallets and an iron drawknife lie neatly arranged on a thick slab of weathered pine. Across the lower third, the workbench surface shows visible tool marks, scattered wood chips, and warm shadows extending toward the bottom edge. In the bottom corners, the rough floor is blanketed in soft curls of blond wood shavings fading into darkness. The lighting is directional natural daylight pouring from a small window on the left, casting crisp shadows across the tool edges and highlighting the grain of the oak timber. The overall composition is balanced and tactile, with an authentic historical palette of wrought iron, warm oak, and natural pine that feels like an early medieval shipyard workshop.",
        "seed": 6024,
        "steps": 25,
        "width": 1024,
        "height": 640
    },
]


def _md_cell(cid: str, text: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def _code_cell(cid: str, code: str) -> dict:
    return {
        "cell_type": "code",
        "id": cid,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build_notebook() -> dict:
    """Assemble the 24-image (8 cores x 3 staged scenes) Kaggle TPU v5e-8 notebook."""
    cell_0_md = """# Neptune Motion: 24-Image Parallel Qwen-Image 2.1 Benchmark on Kaggle TPU v5e-8 (8 × 3)

This standalone notebook executes **24 authentic Neptune Motion production scenes** (`8 TPU cores × 3 staged scenes/core`, **25 denoising steps**, **`1024×640`**) using ComfyUI's uncensored **Qwen-Image 2.1 Native INT8 ConvRot** pipeline:
- **Diffusion Backbone**: `qwen-image-2.1-turbo-int8_convrot.safetensors` (`~6.9 GB`, `abenzerps/Qwen-Image-2.1-Uncensored-Quantized`)
- **Vision-Language Text Encoder**: `qwen3vl_8b_int8_convrot.safetensors` (`~9.3 GB`, `abenzerps/Qwen-Image-2.1-Uncensored-Quantized`)
- **Autoencoder**: `qwen_image_2.1_vae_bf16.safetensors` (`~0.68 GB`, `abenzerps/Qwen-Image-2.1-Uncensored-Quantized`)

### Multi-Core & Memory Architecture:
- **Process-Level Single-Chip Isolation**: Spawns 8 independent `main.py` servers (1 per TPU chip via `TPU_VISIBLE_DEVICES`, `TPU_VISIBLE_CHIPS`, `TPU_CHIPS_PER_HOST_BOUNDS=1,1,1`, disjoint ports `8188–8195`, disjoint `TPU_PROCESS_PORT` & runtime metrics ports, and `tcmalloc` preload) without touching `/dev/vfio` in the parent notebook process.
- **Zero Lock Contention**: Each core uses `--database-url sqlite:///:memory:` and isolated `--user-directory` / `--temp-directory`.
- **Staged Batch DAG per Core**: Encodes all 3 prompts on a core, then samples all 3 in `KSampler`, then decodes via `VAEDecode` — eliminating CLIP ↔ DiT PCIe swaps (`~3×` faster than naive per-image execution).
- **Storage Quota Protection**: Checkpoints stay in `/tmp/ComfyUI` (~17 GB); only the 24 PNGs (~20 MB), `benchmark_report.json`, and server logs are written to `/kaggle/working/output`.

### Kaggle Notebook Settings Before Commit:
1. **Accelerator**: Select **TPU VM v5e-8** (`TpuV5E8`, or GPU/CPU fallback is auto-detected).
2. **Internet**: Toggle **Internet ON**.
3. **Hugging Face Token (Optional)**: Add a Kaggle Secret named `HF_TOKEN` for faster authenticated downloads.
"""

    cell_1_code = """# Step 1: Hardware, Effective RAM & Storage Discovery (Parent Kernel Stays Device-Free)
import os, sys, re, time, json, glob, shutil, subprocess, importlib.util

print(f"Python Version: {sys.version}")

# GPU discovery (nvidia-smi is absent on TPU/CPU VMs)
has_gpu = False
num_gpus = 0
if shutil.which("nvidia-smi"):
    res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if res.returncode == 0:
        gpu_lines = [l for l in res.stdout.splitlines() if l.strip()]
        num_gpus = len(gpu_lines)
        has_gpu = num_gpus > 0
print(f"NVIDIA GPUs Available: {num_gpus} (has_gpu={has_gpu})")
if has_gpu:
    subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])

# TPU chip discovery via device nodes & env without initializing libtpu in parent process
vfio_devs = sorted(glob.glob("/dev/vfio/[0-9]*"), key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 999)
accel_devs = sorted(glob.glob("/dev/accel[0-9]*"))
has_tpu = bool(
    vfio_devs
    or accel_devs
    or "TPU" in os.environ.get("PJRT_DEVICE", "").upper()
    or "TPU_NAME" in os.environ
    or "KAGGLE_TPU_NAME" in os.environ
)
tpu_is_vfio = bool(vfio_devs)
num_tpu_cores = len(vfio_devs) or len(accel_devs) or (8 if has_tpu else 0)
print(f"Google Cloud TPU Hardware: has_tpu={has_tpu}, cores={num_tpu_cores}, vfio={len(vfio_devs)}, accel={len(accel_devs)}")

# Compute effective Host RAM (minimum of /proc/meminfo MemTotal and container cgroup limit)
mem_total_bytes = None
if os.path.exists("/proc/meminfo"):
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemTotal:"):
                mem_total_bytes = int(line.split()[1]) * 1024
                break

cgroup_candidates = [
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
]
cgroup_bytes = None
for cg_path in cgroup_candidates:
    if os.path.exists(cg_path):
        try:
            raw = open(cg_path).read().strip()
            if raw != "max":
                val = int(raw)
                if 0 < val < (1 << 50):
                    cgroup_bytes = val if cgroup_bytes is None else min(cgroup_bytes, val)
        except Exception:
            pass

effective_ram_bytes = min(x for x in (mem_total_bytes, cgroup_bytes) if x is not None) if (mem_total_bytes or cgroup_bytes) else (32 * 1024**3)
effective_ram_gb = effective_ram_bytes / (1024 ** 3)
print(f"Effective Host RAM: {effective_ram_gb:.1f} GB (cgroup_capped={cgroup_bytes is not None and mem_total_bytes is not None and cgroup_bytes < mem_total_bytes})")

# Keep ~17 GB checkpoints in /tmp/ComfyUI; keep persistent outputs < 50 MB in /kaggle/working/output
COMFY_DIR = "/tmp/ComfyUI"
WORK_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "/tmp/kaggle_work"
OUTPUT_DIR = os.path.join(WORK_DIR, "output")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

tmp_free_gb = shutil.disk_usage("/tmp").free / (1024 ** 3)
work_free_gb = shutil.disk_usage(WORK_DIR).free / (1024 ** 3)
print(f"Disk Free -> /tmp: {tmp_free_gb:.1f} GB | {WORK_DIR}: {work_free_gb:.1f} GB")
"""

    cell_2_code = """# Step 2: Clone ComfyUI, Install Dependencies & Concurrent Multi-Chip TPU Pre-Flight Probe
if not os.path.exists(os.path.join(COMFY_DIR, ".git")):
    print("Cloning ComfyUI (feat/tpu-multi-core-support)...")
    subprocess.run([
        "git", "clone", "--depth", "1", "-b", "feat/tpu-multi-core-support",
        "https://github.com/vuthanhtrung2010/ComfyUI.git", COMFY_DIR
    ], check=True)
else:
    print(f"ComfyUI already present at {COMFY_DIR}; updating...")
    subprocess.run(["git", "-C", COMFY_DIR, "pull", "--ff-only"], check=False)

# Install PyTorch CPU + torch_tpu when on TPU if missing
if has_tpu and importlib.util.find_spec("torch_tpu") is None:
    print("Installing PyTorch CPU wheel and torch_tpu...")
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "-U",
        "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cpu"
    ], check=True)
    subprocess.run([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "torch_tpu"
    ], check=True)

# Install ComfyUI requirements while preserving preinstalled/TPU torch packages (keep torchsde!)
print("Installing ComfyUI dependencies...")
skip_torch = has_tpu or (importlib.util.find_spec("torch") is not None)
with open(os.path.join(COMFY_DIR, "requirements.txt"), "r") as f:
    reqs = []
    for raw_line in f:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkg_name = re.split(r"[=<>!~\\[\\]; ]", line, maxsplit=1)[0].strip().lower()
        if skip_torch and pkg_name in ("torch", "torchvision", "torchaudio"):
            continue
        reqs.append(line)

req_tmp = "/tmp/comfy_reqs_kaggle.txt"
with open(req_tmp, "w") as f:
    f.write("\\n".join(reqs) + "\\n")

subprocess.run([
    sys.executable, "-m", "pip", "install", "-q", "-r", req_tmp, "gguf"
], check=True)

# Multi-chip single-core subprocess probe (verifies concurrent libtpu isolation before 17GB downloads)
TPU_USE_V5LITEPOD1 = tpu_is_vfio

def _build_tpu_child_env(core_id, use_v5litepod1):
    env = os.environ.copy()
    base_metric = 8431 + core_id * 4
    env.update({
        "TPU_VISIBLE_DEVICES": str(core_id),
        "TPU_VISIBLE_CHIPS": str(core_id),
        "TPU_CHIPS_PER_HOST_BOUNDS": "1,1,1",
        "TPU_HOST_BOUNDS": "1,1,1",
        "TPU_WORKER_ID": "0",
        "TPU_WORKER_HOSTNAMES": "localhost",
        "TPU_SKIP_MDS_QUERY": "1",
        "PJRT_DEVICE": "TPU",
        "TPU_PROCESS_PORT": str(8476 + core_id),
        "TPU_RUNTIME_METRICS_PORTS": f"{base_metric},{base_metric+1},{base_metric+2},{base_metric+3}",
        "OMP_NUM_THREADS": "4",
        "PYTHONWARNINGS": "ignore",
    })
    if use_v5litepod1:
        env["TPU_ACCELERATOR_TYPE"] = "v5litepod-1"
    else:
        env.pop("TPU_ACCELERATOR_TYPE", None)
    tcmalloc_path = "/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4"
    if os.path.exists(tcmalloc_path):
        env["LD_PRELOAD"] = tcmalloc_path
    return env

def _run_multichip_probe(use_v5litepod1):
    probe_cores = sorted({0, min(1, max(0, num_tpu_cores - 1))})
    probe_py = (
        "import torch, torch_tpu\\n"
        "try: torch.utils.rename_privateuse1_backend('tpu'); torch.utils.generate_methods_for_privateuse1_backend()\\n"
        "except Exception: pass\\n"
        "if not hasattr(torch, 'tpu') and hasattr(torch, '_register_device_module'): torch._register_device_module('tpu', torch_tpu)\\n"
        "dev = torch.device('tpu')\\n"
        "a = torch.randn((16, 16), device=dev, dtype=torch.bfloat16)\\n"
        "c = (a @ a).cpu()\\n"
        "free_b, tot_b = torch.tpu.get_memory_info(dev) if hasattr(torch, 'tpu') and hasattr(torch.tpu, 'get_memory_info') else (0, 0)\\n"
        "print(f'PROBE_OK shape={tuple(c.shape)} hbm_total_gb={tot_b/1e9:.2f} free_gb={free_b/1e9:.2f}')\\n"
    )
    procs = []
    for cid in probe_cores:
        p = subprocess.Popen(
            [sys.executable, "-c", probe_py],
            env=_build_tpu_child_env(cid, use_v5litepod1),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        procs.append((cid, p))
    all_ok = True
    for cid, p in procs:
        out, err = p.communicate(timeout=90)
        line = (out.strip().splitlines()[-1] if out.strip() else err.strip().splitlines()[-1] if err.strip() else "no output")
        print(f"  [Probe Core {cid}, v5litepod-1={use_v5litepod1}] rc={p.returncode}: {line}")
        if p.returncode != 0 or "PROBE_OK" not in out:
            all_ok = False
    return all_ok

if has_tpu:
    print("Running concurrent 2-core TPU pre-flight probe...")
    if not _run_multichip_probe(TPU_USE_V5LITEPOD1):
        print("Retrying probe with opposite TPU_ACCELERATOR_TYPE setting...")
        TPU_USE_V5LITEPOD1 = not TPU_USE_V5LITEPOD1
        if not _run_multichip_probe(TPU_USE_V5LITEPOD1):
            raise RuntimeError("TPU multi-chip subprocess probe failed! Verify TPU VM accelerator is attached.")
    print(f"TPU multi-chip isolation verified! TPU_USE_V5LITEPOD1={TPU_USE_V5LITEPOD1}")
"""

    cell_3_code = """# Step 3: Download Uncensored Qwen-Image 2.1 Native INT8 Checkpoints to /tmp/ComfyUI
def _resolve_hf_token():
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        try:
            from kaggle_secrets import UserSecretsClient
            tok = UserSecretsClient().get_secret("HF_TOKEN")
        except Exception:
            tok = None
    return tok.strip() if isinstance(tok, str) and tok.strip() else None

hf_token = _resolve_hf_token()
use_aria2 = shutil.which("aria2c") is not None
print(f"Download setup -> tool: {'aria2c (8x)' if use_aria2 else 'curl'}, auth: {'HF_TOKEN active' if hf_token else 'anonymous (add HF_TOKEN secret if needed)'}")

def download_checkpoint(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        print(f"[CACHE HIT] {os.path.basename(dest)} ({os.path.getsize(dest) / 1e6:.1f} MB)")
        return
    print(f"[DOWNLOADING] {os.path.basename(dest)}...")
    t0 = time.time()
    if use_aria2:
        cmd = [
            "aria2c", "-x", "8", "-s", "8", "-k", "1M", "--continue=true",
            "--console-log-level=warn", "-d", os.path.dirname(dest), "-o", os.path.basename(dest),
        ]
        if hf_token:
            cmd += ["--header", f"Authorization: Bearer {hf_token}"]
        cmd.append(url)
    else:
        cmd = ["curl", "-L", "-C", "-", "--retry", "5", "-sS", "-o", dest]
        if hf_token:
            cmd += ["-H", f"Authorization: Bearer {hf_token}"]
        cmd.append(url)
    subprocess.run(cmd, check=True)
    dt = max(time.time() - t0, 1e-3)
    size_mb = os.path.getsize(dest) / 1e6
    print(f"[COMPLETE] {os.path.basename(dest)}: {size_mb:.1f} MB in {dt:.1f}s ({size_mb / dt:.1f} MB/s)")

models = [
    ("models/unet/qwen-image-2.1-turbo-int8_convrot.safetensors",
     "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/qwen-image-2.1-turbo-int8_convrot.safetensors"),
    ("models/clip/qwen3vl_8b_int8_convrot.safetensors",
     "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/text_encoders/qwen3vl_8b_int8_convrot.safetensors"),
    ("models/vae/qwen_image_2.1_vae_bf16.safetensors",
     "https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-Quantized/resolve/main/vae/qwen_image_2.1_vae_bf16.safetensors"),
]

for rel_path, url in models:
    download_checkpoint(url, os.path.join(COMFY_DIR, rel_path))

print("All 3 Uncensored Qwen-Image 2.1 checkpoints verified in /tmp/ComfyUI/models!")
"""

    prompts_literal = json.dumps(PROD_PROMPTS, indent=2)
    cell_4_code = f"""# Step 4: 24 Authentic Neptune Motion Production Prompts & Staged Batch DAG Builder
PROD_PROMPTS = {prompts_literal}

def build_staged_batch_workflow(shard_prompts, shard_idx):
    \"\"\"Build a single DAG forcing: all CLIP encodes -> all KSamplers -> all VAEDecodes -> SaveImages.\"\"\"
    wf = {{
        "unet": {{"class_type": "UNETLoader", "inputs": {{"unet_name": "qwen-image-2.1-turbo-int8_convrot.safetensors", "weight_dtype": "default"}}}},
        "clip": {{"class_type": "CLIPLoader", "inputs": {{"clip_name": "qwen3vl_8b_int8_convrot.safetensors", "type": "qwen_image", "device": "default"}}}},
        "vae": {{"class_type": "VAELoader", "inputs": {{"vae_name": "qwen_image_2.1_vae_bf16.safetensors"}}}},
        "latent": {{"class_type": "EmptyLatentImage", "inputs": {{"width": 1024, "height": 640, "batch_size": 1}}}},
    }}

    for p in shard_prompts:
        i = p["id"]
        wf[f"enc{{i}}"] = {{"class_type": "CLIPTextEncode", "inputs": {{"text": p["prompt"], "clip": ["clip", 0]}}}}
        wf[f"zero{{i}}"] = {{"class_type": "ConditioningZeroOut", "inputs": {{"conditioning": [f"enc{{i}}", 0]}}}}

    ids = [p["id"] for p in shard_prompts]
    gate_node = f"zero{{ids[0]}}"
    for idx in range(1, len(ids)):
        next_gate = f"gate_{{idx}}"
        wf[next_gate] = {{
            "class_type": "ConditioningAverage",
            "inputs": {{
                "conditioning_to": [gate_node, 0],
                "conditioning_from": [f"zero{{ids[idx]}}", 0],
                "conditioning_to_strength": 1.0,
            }},
        }}
        gate_node = next_gate

    for p in shard_prompts:
        i = p["id"]
        wf[f"ks{{i}}"] = {{
            "class_type": "KSampler",
            "inputs": {{
                "model": ["unet", 0],
                "seed": p["seed"],
                "steps": p["steps"],
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "positive": [f"enc{{i}}", 0],
                "negative": [gate_node, 0],
                "latent_image": ["latent", 0],
                "denoise": 1.0,
            }},
        }}

    batch_node = f"ks{{ids[0]}}"
    for idx in range(1, len(ids)):
        next_batch = f"lbatch_{{idx}}"
        wf[next_batch] = {{
            "class_type": "LatentBatch",
            "inputs": {{"samples1": [batch_node, 0], "samples2": [f"ks{{ids[idx]}}", 0]}},
        }}
        batch_node = next_batch

    for idx, p in enumerate(shard_prompts):
        i = p["id"]
        wf[f"lfb{{i}}"] = {{
            "class_type": "LatentFromBatch",
            "inputs": {{"samples": [batch_node, 0], "batch_index": idx, "length": 1}},
        }}
        wf[f"dec{{i}}"] = {{
            "class_type": "VAEDecode",
            "inputs": {{"samples": [f"lfb{{i}}", 0], "vae": ["vae", 0]}},
        }}
        wf[f"save{{i}}"] = {{
            "class_type": "SaveImage",
            "inputs": {{"images": [f"dec{{i}}", 0], "filename_prefix": f"neptune25_s{{shard_idx:02d}}_{{i:02d}}_{{p['name']}}"}},
        }}
    return wf

print(f"Loaded {{len(PROD_PROMPTS)}} Neptune Motion production prompts across {{len({{p['video'] for p in PROD_PROMPTS}})}} videos.")
"""

    cell_5_code = """# Step 5: Parallel Multi-Core Cluster Execution (8 TPU Cores x 3 Staged Images/Core = 24 Images)
import signal, threading, concurrent.futures, urllib.request, urllib.error

if has_tpu:
    backend = "tpu"
    num_workers = num_tpu_cores
    max_concurrent = min(num_workers, max(1, int((effective_ram_gb - 8.0) // 12.5)))
elif has_gpu:
    backend = "gpu"
    num_workers = max(1, num_gpus)
    max_concurrent = num_workers
else:
    backend = "cpu"
    num_workers = 1
    max_concurrent = 1

# Contiguous shards: on 8 cores, Core 0..7 each own one video's 3 scenes
shards = [
    PROD_PROMPTS[i * len(PROD_PROMPTS) // num_workers : (i + 1) * len(PROD_PROMPTS) // num_workers]
    for i in range(num_workers)
]
waves = [list(range(start, min(start + max_concurrent, num_workers))) for start in range(0, num_workers, max_concurrent)]

print(f"Cluster Plan -> backend={backend.upper()}, total_workers={num_workers}, max_concurrent={max_concurrent}, waves={len(waves)}, total_images={len(PROD_PROMPTS)}")

def _build_worker_launch(worker_id, port):
    user_dir = f"/tmp/comfy_user_core_{worker_id}"
    temp_dir = f"/tmp/comfy_temp_core_{worker_id}"
    os.makedirs(user_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)
    cmd = [
        sys.executable, os.path.join(COMFY_DIR, "main.py"),
        "--port", str(port),
        "--listen", "127.0.0.1",
        "--database-url", "sqlite:///:memory:",
        "--user-directory", user_dir,
        "--temp-directory", temp_dir,
        "--output-directory", OUTPUT_DIR,
        "--verbose", "INFO",
    ]
    if backend == "tpu":
        env = _build_tpu_child_env(worker_id, TPU_USE_V5LITEPOD1)
        cmd += ["--tpu-device", str(worker_id)]
    elif backend == "gpu":
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(worker_id)
        cmd += ["--lowvram"]
    else:
        env = os.environ.copy()
        cmd += ["--cpu"]
    return cmd, env

progress_lock = threading.Lock()
progress_state = {"done_images": 0, "done_shards": 0}

def _run_shard_on_port(shard_idx, shard_prompts, port):
    # Chunk into sub-batches of <=3 images when running many images on few workers (e.g. 1-GPU fallback)
    sub_batches = [shard_prompts[k:k+3] for k in range(0, len(shard_prompts), 3)]
    shard_start = time.time()
    files = []
    sub_durations = []
    for sb in sub_batches:
        wf = build_staged_batch_workflow(sb, shard_idx)
        payload = json.dumps({"prompt": wf, "client_id": f"kaggle_core_{shard_idx}"}).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{port}/prompt", data=payload, headers={"Content-Type": "application/json"})
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=30) as resp:
            prompt_id = json.loads(resp.read().decode())["prompt_id"]

        sb_done = False
        while time.time() - t0 < 1200:
            time.sleep(2.0)
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/history/{prompt_id}", timeout=10) as hresp:
                    history = json.loads(hresp.read().decode())
                if prompt_id in history:
                    entry = history[prompt_id]
                    st = entry.get("status", {})
                    if st.get("completed", False):
                        for node_out in entry.get("outputs", {}).values():
                            for img in node_out.get("images", []):
                                files.append(img["filename"])
                        sub_durations.append(round(time.time() - t0, 2))
                        with progress_lock:
                            progress_state["done_images"] += len(sb)
                        sb_done = True
                        break
                    elif st.get("status_str") == "error":
                        raise RuntimeError(f"Core {shard_idx} workflow failed: {st}")
            except urllib.error.URLError:
                pass
        if not sb_done:
            raise TimeoutError(f"Core {shard_idx} timed out after 1200s on prompt {prompt_id}")

    with progress_lock:
        progress_state["done_shards"] += 1
    return {
        "core_id": shard_idx,
        "port": port,
        "num_images": len(shard_prompts),
        "duration_seconds": round(time.time() - shard_start, 2),
        "sub_batch_durations_s": sub_durations,
        "output_files": sorted(files),
        "status": "success",
    }

subprocess.run(["pkill", "-f", "ComfyUI/main.py"], capture_output=True)
time.sleep(1.0)

overall_start = time.time()
shard_results = []

for w_idx, wave_cores in enumerate(waves):
    print(f"\\n=== Wave {w_idx+1}/{len(waves)}: Launching {len(wave_cores)} {backend.upper()} servers {wave_cores} ===")
    active = []
    try:
        for slot_idx, core_id in enumerate(wave_cores):
            port = 8188 + slot_idx
            cmd, env = _build_worker_launch(core_id, port)
            log_path = os.path.join(LOG_DIR, f"instance_core_{core_id}.log")
            lf = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, cwd=COMFY_DIR, env=env,
                stdin=subprocess.DEVNULL, stdout=lf, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            lf.close()
            active.append({"core_id": core_id, "port": port, "proc": proc, "log_path": log_path})
            time.sleep(1.0)

        # Wait up to 150s for every server in this wave to accept HTTP requests
        pending = {item["core_id"]: item for item in active}
        t_wait = time.time()
        while pending and (time.time() - t_wait) < 150:
            for cid in list(pending.keys()):
                item = pending[cid]
                if item["proc"].poll() is not None:
                    tail = "".join(open(item["log_path"], errors="replace").readlines()[-30:])
                    raise RuntimeError(f"Core {cid} exited early (code={item['proc'].returncode}):\\n{tail}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{item['port']}/system_stats", timeout=2) as r:
                        if r.status == 200:
                            print(f"  [READY] Core {cid} online on port {item['port']}")
                            del pending[cid]
                except Exception:
                    pass
            if pending:
                time.sleep(2.0)

        if pending:
            c0 = next(iter(pending.values()))
            tail = "".join(open(c0["log_path"], errors="replace").readlines()[-30:])
            raise TimeoutError(f"Timed out waiting for cores {list(pending.keys())}. Sample log:\\n{tail}")

        print(f"Dispatching {sum(len(shards[c]) for c in wave_cores)} images concurrently across {len(active)} servers...")
        stop_monitor = threading.Event()
        def _monitor_loop():
            while not stop_monitor.wait(15.0):
                with progress_lock:
                    di, ds = progress_state["done_images"], progress_state["done_shards"]
                print(f"  [PROGRESS] {int(time.time() - overall_start)}s elapsed | {di}/{len(PROD_PROMPTS)} images complete ({ds}/{num_workers} cores done)", flush=True)

        mon_t = threading.Thread(target=_monitor_loop, daemon=True)
        mon_t.start()
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(active)) as pool:
                futs = {
                    pool.submit(_run_shard_on_port, item["core_id"], shards[item["core_id"]], item["port"]): item["core_id"]
                    for item in active
                }
                for fut in concurrent.futures.as_completed(futs):
                    res = fut.result()
                    shard_results.append(res)
                    print(f"  [CORE {res['core_id']} DONE] {res['num_images']} images in {res['duration_seconds']:.1f}s -> {res['output_files']}")
        finally:
            stop_monitor.set()
            mon_t.join(timeout=2.0)
    finally:
        for item in active:
            proc = item["proc"]
            if proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
        for item in active:
            proc = item["proc"]
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:
                    proc.kill()

wall_clock_seconds = round(time.time() - overall_start, 2)
shard_results.sort(key=lambda r: r["core_id"])

# Parse steady-state sampling speeds from per-core server logs
per_core_steady_step_s = {}
for cid in range(num_workers):
    lp = os.path.join(LOG_DIR, f"instance_core_{cid}.log")
    if os.path.exists(lp):
        txt = open(lp, errors="replace").read()
        speeds = []
        for m in re.finditer(r"25/25\\s*\\[[^,\\]]+,\\s*([0-9.]+)(s/it|it/s)\\]", txt):
            v = float(m.group(1))
            speeds.append(round(v if m.group(2) == "s/it" else (1.0 / v if v > 0 else 0.0), 3))
        if speeds:
            per_core_steady_step_s[str(cid)] = speeds

all_steady = [s for vals in per_core_steady_step_s.values() for s in vals]
# Last reported rate of each 25/25 progress bar is overall average; first steady bar or per-core average steps/s
last_rates = [vals[-1] for vals in per_core_steady_step_s.values() if vals]
cluster_agg_steps_per_sec = round(sum(1.0 / s for s in last_rates if s > 0), 2) if last_rates else None

successful_images = sum(len(r["output_files"]) for r in shard_results if r["status"] == "success")
eff_sec_per_img = round(wall_clock_seconds / successful_images, 2) if successful_images else None
throughput_ipm = round(successful_images / (wall_clock_seconds / 60.0), 2) if wall_clock_seconds > 0 else 0.0

benchmark_report = {
    "benchmark_name": "Neptune Motion 24-Image Parallel Kaggle Benchmark (Qwen-Image 2.1 Uncensored INT8 ConvRot)",
    "backend": backend,
    "num_workers": num_workers,
    "max_concurrent_workers": max_concurrent,
    "waves": len(waves),
    "total_prompts": len(PROD_PROMPTS),
    "successful_images": successful_images,
    "steps_per_image": 25,
    "resolution": "1024x640",
    "wall_clock_seconds": wall_clock_seconds,
    "effective_seconds_per_image": eff_sec_per_img,
    "throughput_images_per_minute": throughput_ipm,
    "cluster_aggregate_steps_per_sec": cluster_agg_steps_per_sec,
    "per_core_step_speeds_s": per_core_steady_step_s,
    "shards": shard_results,
}

report_path = os.path.join(OUTPUT_DIR, "benchmark_report.json")
with open(report_path, "w") as f:
    json.dump(benchmark_report, f, indent=2)

print("\\n==========================================================================")
print(f"COMPLETED {successful_images}/{len(PROD_PROMPTS)} images in {wall_clock_seconds:.1f}s ({wall_clock_seconds/60:.2f} min)!")
print(f"Effective amortized latency: {eff_sec_per_img}s / image | Throughput: {throughput_ipm} img/min")
print(f"Report saved to: {report_path}")
print("==========================================================================")
"""

    cell_6_code = """# Step 6: Verify All 24 Generated Images & Display Contact Sheet Gallery
from PIL import Image, ImageStat
from IPython.display import Image as IPyImage, display

png_files = sorted(f for f in os.listdir(OUTPUT_DIR) if f.startswith("neptune25_s") and f.endswith(".png"))
print(f"Verified PNG count in {OUTPUT_DIR}: {len(png_files)} / {len(PROD_PROMPTS)}")

bad_images = []
for fname in png_files:
    fpath = os.path.join(OUTPUT_DIR, fname)
    with Image.open(fpath) as im:
        w, h = im.size
        std = max(ImageStat.Stat(im.convert("RGB")).stddev)
        if (w, h) != (1024, 640) or std < 1.0:
            bad_images.append((fname, (w, h), round(std, 2)))

if bad_images:
    print(f"WARNING: Found {len(bad_images)} invalid/blank images: {bad_images}")
else:
    print("ALL generated PNGs passed integrity check (1024x640, non-blank pixel variance).")

print("\\n=== Per-Core Shard Summary ===")
for s in benchmark_report["shards"]:
    print(f"  Core {s['core_id']}: {s['num_images']} imgs | {s['duration_seconds']:.1f}s | {s['output_files']}")

if len(png_files) < len(PROD_PROMPTS):
    raise RuntimeError(f"Expected {len(PROD_PROMPTS)} images, got {len(png_files)}")

# Build compact contact sheet in WORK_DIR for quick visual inspection
cols, rows = 4, (len(png_files) + 3) // 4
thumb_w, thumb_h = 512, 320
grid = Image.new("RGB", (cols * thumb_w, rows * thumb_h), (245, 235, 215))
for idx, fname in enumerate(png_files):
    with Image.open(os.path.join(OUTPUT_DIR, fname)) as im:
        thumb = im.convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    r, c = divmod(idx, cols)
    grid.paste(thumb, (c * thumb_w, r * thumb_h))

grid_path = os.path.join(WORK_DIR, "preview_grid_24.png")
grid.save(grid_path)
print(f"Saved 24-image contact sheet to: {grid_path}")
display(IPyImage(filename=grid_path))
"""

    nb = {
        "cells": [
            _md_cell("cell-0", cell_0_md),
            _code_cell("cell-1", cell_1_code),
            _code_cell("cell-2", cell_2_code),
            _code_cell("cell-3", cell_3_code),
            _code_cell("cell-4", cell_4_code),
            _code_cell("cell-5", cell_5_code),
            _code_cell("cell-6", cell_6_code),
        ],
        "metadata": {
            "accelerator": "TPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main():
    nb = build_notebook()
    targets = [
        "kaggle/qwen_image_2_1_kaggle.ipynb",
        "kaggle/tpu_comfyui_benchmark/comfyui_tpu_benchmark.ipynb",
    ]
    for path in targets:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully generated {path} ({len(PROD_PROMPTS)} prompts, {len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
