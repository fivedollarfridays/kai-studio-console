"""Every third-party asset bundled in ``console/`` -- a font file of any
extension, a license banner in any bundled JS, or a vendored JS library --
has a corresponding entry in
``THIRD_PARTY_NOTICES.md``, and that file carries the SIL OFL 1.1 license
text verbatim, as the OFL requires it to travel with the Fraunces and
JetBrains Mono fonts.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONSOLE = ROOT / "console"
ASSETS = CONSOLE / "assets"
NOTICES = ROOT / "THIRD_PARTY_NOTICES.md"

# A vendored JS library is detected by a byte signature in the committed
# bundle, mapped to the name its notices entry must carry. GSAP ships
# embedded in the bundle, not as a standalone file under assets/.
VENDORED_JS_LIBRARIES = {
    re.compile(rb"gsap", re.IGNORECASE): "GSAP",
}

SIL_OFL_1_1_TEXT = """\
-----------------------------------------------------------
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007
-----------------------------------------------------------

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
"""


# Every font file of ANY extension counts, not only the fontsource woff/woff2
# shapes this build happens to emit today.
FONT_EXTENSIONS = {".ttf", ".otf", ".woff", ".woff2"}

# A license banner in bundled JS: an esbuild-preserved `/*! ... */` legal
# comment, any other block comment carrying `@license`, or a `//` line that
# does. Generic on purpose -- the scan must catch libraries nobody listed.
BANNER = re.compile(
    rb"/\*!.*?\*/|/\*(?:(?!\*/).)*?@license.*?\*/|//[^\n]*@license[^\n]*", re.DOTALL
)

_VERSION_TAIL = re.compile(r"\s+v?\d[\w.]*$")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _notice_entries(notices: str) -> set[str]:
    """Each ``## <Name> [version]`` heading's normalized name; the license
    text sections (``## ... License ...``) name no bundled component."""
    names = set()
    for line in notices.splitlines():
        if line.startswith("## ") and "license" not in line.lower():
            names.add(_norm(_VERSION_TAIL.sub("", line[3:].strip())))
    return names


def _font_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in FONT_EXTENSIONS)


def _fonts_without_entry(fonts: list[Path], notices: str) -> list[str]:
    """A font file is covered when its name starts with a notices entry's
    name (``jetbrains-mono-latin-400-normal-X.woff2`` -> ``JetBrains Mono``)."""
    entries = _notice_entries(notices)
    return sorted(f.name for f in fonts if not any(_norm(f.name).startswith(e) for e in entries))


def _js_bundles(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix in {".js", ".mjs"})


def _banners_without_entry(bundle: bytes, notices: str) -> list[str]:
    """Every license banner whose text names no notices entry."""
    entries = _notice_entries(notices)
    missing = []
    for m in BANNER.finditer(bundle):
        text = m.group(0).decode("utf-8", "replace")
        if not any(e in _norm(text) for e in entries):
            missing.append(text)
    return missing


def _missing_js_libraries(bundle: bytes, notices: str, catalog: dict) -> list[str]:
    lowered = notices.lower()
    missing = []
    for pattern, name in catalog.items():
        if pattern.search(bundle) and name.lower() not in lowered:
            missing.append(name)
    return missing


def _notices_text() -> str:
    return NOTICES.read_text(encoding="utf-8")


def test_notices_file_exists():
    assert NOTICES.is_file()


def test_every_bundled_font_file_has_a_notices_entry():
    fonts = _font_files(CONSOLE)
    assert fonts, "no font files found under console/"
    assert _fonts_without_entry(fonts, _notices_text()) == []


def test_the_font_scanner_catches_a_planted_font_of_any_extension(tmp_path):
    # red-first proof: an unlisted family, and a .ttf/.otf the old
    # woff-only, fontsource-shape-only scan never looked at
    for name in (
        "jetbrains-mono-latin-400-normal-AAAA.woff2",
        "comic-sans-latin-400-normal-ZZZZZZZZ.woff2",
        "Lato-Regular.ttf",
        "SourceSerif.otf",
    ):
        (tmp_path / name).write_bytes(b"font")
    missing = _fonts_without_entry(_font_files(tmp_path), "## JetBrains Mono\n")
    assert missing == [
        "Lato-Regular.ttf",
        "SourceSerif.otf",
        "comic-sans-latin-400-normal-ZZZZZZZZ.woff2",
    ]


def test_every_license_banner_in_bundled_js_has_a_notices_entry():
    bundles = _js_bundles(CONSOLE)
    assert bundles, "no JS bundle found to scan"
    bundle = b"\n".join(p.read_bytes() for p in bundles)
    assert _banners_without_entry(bundle, _notices_text()) == []


def test_the_banner_scanner_catches_planted_unlisted_banners():
    # red-first proof: libraries no catalog in this file names
    planted = (
        b"var a=1;/*! lodash v4.17.21 | (c) OpenJS Foundation | MIT */var b=2;"
        b"/** @license React v18.2.0 */var c=3;"
        b"// @license MIT left-pad\n"
        b"/*! GSAP 3.15.0 */"
    )
    missing = _banners_without_entry(planted, "## GSAP 3.15.0\n")
    assert [m.split()[1] if m.startswith("/*") else m for m in missing] == [
        "lodash",
        "@license",
        "// @license MIT left-pad",
    ]


def test_every_vendored_js_library_has_a_notices_entry():
    bundle = b"\n".join(p.read_bytes() for p in _js_bundles(CONSOLE))
    assert bundle, "no JS bundle found to scan for vendored libraries"
    assert _missing_js_libraries(bundle, _notices_text(), VENDORED_JS_LIBRARIES) == []


def test_the_library_scanner_catches_a_planted_unlisted_library():
    # assembled at runtime so this file itself never names the planted library
    planted_bundle = b"import " + b"gsap" + b" from 'gsap'"
    missing = _missing_js_libraries(
        planted_bundle, "no mention of that lib here.", VENDORED_JS_LIBRARIES
    )
    assert missing == ["GSAP"]


def test_ofl_license_text_is_present_verbatim():
    assert SIL_OFL_1_1_TEXT in _notices_text()


def test_notices_names_gsap_version():
    assert "3.15.0" in _notices_text()
