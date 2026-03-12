#!/usr/bin/env python3
# Copyright (c) 2026 Vistralis Labs. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build API documentation as static HTML pages.

Extracts module structure from quak using pdoc's programmatic API,
then renders multi-page HTML with the Vistralis documentation theme.

Usage:
    python docs/build_api_docs.py
    python docs/build_api_docs.py --output /path/to/website/quakfilter/api/
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any

import pdoc.doc


# ---------------------------------------------------------------------------
# Page definitions — which classes/functions go on which page
# ---------------------------------------------------------------------------

ENUM_CLASSES = {"TimeMode"}

# Classes whose members are mostly inherited — only show own methods
THIN_CLASSES: set[str] = set()

PAGE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "slug": "filters",
        "title": "Filters",
        "subtitle": "UKF and IMM estimators for non-linear state estimation",
        "classes": ["UKF", "IMMEstimator"],
        "functions": [],
    },
    {
        "slug": "models",
        "title": "Motion Models",
        "subtitle": "Rigid body dynamics — constant velocity, acceleration, turn-rate, Singer, helical",
        "classes": [
            "UKFModel",
            "SystemModel",
            "RigidBodyCVModel",
            "RigidBodyCAModel",
            "RigidBodyCTRVModel",
            "RigidBodySingerModel",
            "RigidBodyHelicalModel",
            "ProjectiveBBoxModel",
        ],
        "functions": [],
    },
    {
        "slug": "sensors",
        "title": "Sensors",
        "subtitle": "Measurement models for pose, velocity, and projective bounding box sensors",
        "classes": [
            "MeasurementModel",
            "RigidBodyPoseSensorModel",
            "PoseSensor",
            "VelocitySensor",
            "ProjectiveBBoxSensor",
            "CameraParams",
            "NoiseModel",
        ],
        "functions": [],
    },
    {
        "slug": "tracking",
        "title": "Tracking",
        "subtitle": "Multi-object track pool with birth, death, and data association",
        "classes": ["TrackPool", "StateEstimate", "TimeMode"],
        "functions": [],
    },
    {
        "slug": "utilities",
        "title": "Utilities",
        "subtitle": "Manifolds, result types, and time conversions",
        "classes": [
            "Manifold",
            "Euclidean",
            "Quaternion",
            "UKFPredictionResult",
        ],
        "functions": [
            "to_epoch_seconds",
            "from_epoch_seconds",
            "to_duration_seconds",
        ],
    },
]


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def extract_docstring(member: Any) -> str:
    """Extract and clean a docstring from a pdoc member."""
    doc = member.docstring or ""
    return doc.strip()


def extract_signature(member: pdoc.doc.Function) -> str:
    """Extract a clean, single-line function/method signature string."""
    try:
        sig = str(member.signature)
        sig = sig.replace("numpy.", "np.")
        sig = sig.replace("typing.", "")
        sig = sig.replace("<class 'numpy.float64'>", "np.float64")
        # Collapse to single line
        sig = " ".join(sig.split())
        return sig
    except Exception:
        return "()"


def _get_own_methods(cls_obj: type) -> set[str]:
    """Get methods defined directly on this class (not inherited)."""
    own = set()
    for name in cls_obj.__dict__:
        if not name.startswith("_") or name in ("__init__", "__mul__", "__rmul__"):
            own.add(name)
    return own


def extract_class(cls: pdoc.doc.Class) -> dict[str, Any]:
    """Extract a class into a structured dict."""
    methods = []
    properties = []

    try:
        own_methods = _get_own_methods(cls.obj)
    except Exception:
        own_methods = None

    is_enum = cls.name in ENUM_CLASSES
    is_thin = cls.name in THIN_CLASSES

    for name, member in cls.members.items():
        if name.startswith("_") and name not in ("__init__", "__mul__", "__rmul__"):
            continue

        if isinstance(member, pdoc.doc.Function):
            is_own = own_methods is None or name in own_methods
            methods.append(
                {
                    "name": name,
                    "signature": extract_signature(member),
                    "docstring": extract_docstring(member),
                    "is_classmethod": name.startswith("from_"),
                    "is_staticmethod": False,
                    "is_own": is_own,
                }
            )
        elif isinstance(member, pdoc.doc.Variable):
            properties.append(
                {
                    "name": name,
                    "docstring": extract_docstring(member),
                }
            )

    return {
        "name": cls.name,
        "qualname": cls.qualname,
        "docstring": extract_docstring(cls),
        "methods": methods,
        "properties": properties,
        "bases": [b.replace("quak.", "") for b in _get_bases(cls)],
        "is_enum": is_enum,
        "is_thin": is_thin,
    }


def _get_bases(cls: pdoc.doc.Class) -> list[str]:
    """Get base class names."""
    try:
        obj = cls.obj
        return [
            b.__module__ + "." + b.__qualname__
            for b in obj.__bases__
            if b is not object
        ]
    except Exception:
        return []


def extract_function(func: pdoc.doc.Function) -> dict[str, Any]:
    """Extract a module-level function."""
    return {
        "name": func.name,
        "signature": extract_signature(func),
        "docstring": extract_docstring(func),
    }


def extract_module() -> dict[str, Any]:
    """Extract the full quak module structure."""
    mod = pdoc.doc.Module.from_name("quak")
    version = ""
    try:
        import quak

        version = quak.__version__
    except Exception:
        pass

    classes = {}
    functions = {}

    for name, member in mod.members.items():
        if isinstance(member, pdoc.doc.Class):
            classes[name] = extract_class(member)
        elif isinstance(member, pdoc.doc.Function):
            functions[name] = extract_function(member)

    return {
        "version": version,
        "docstring": extract_docstring(mod),
        "classes": classes,
        "functions": functions,
    }


# ---------------------------------------------------------------------------
# Docstring → structured HTML
# ---------------------------------------------------------------------------


def _parse_google_docstring(doc: str) -> dict[str, Any]:
    """Parse a Google-style docstring into structured sections.

    Returns:
        Dict with keys: summary, description, args, returns, raises, notes, examples
    """
    if not doc:
        return {"summary": "", "sections": []}

    lines = doc.split("\n")
    result: dict[str, Any] = {"summary": "", "sections": []}

    # First paragraph = summary
    summary_lines = []
    i = 0
    while i < len(lines) and lines[i].strip():
        summary_lines.append(lines[i].strip())
        i += 1
    result["summary"] = " ".join(summary_lines)

    # Parse remaining sections
    current_section = None
    current_items: list[dict[str, str]] = []
    current_text: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        section_match = re.match(
            r"^(Args|Parameters|Returns|Return type|Raises|Note|Example|"
            r"Attributes|Models|Variables|Warning):\s*$",
            stripped,
        )
        if section_match:
            if current_section:
                result["sections"].append(
                    {
                        "type": current_section,
                        "items": current_items,
                        "text": "\n".join(current_text).strip(),
                    }
                )
            current_section = section_match.group(1)
            current_items = []
            current_text = []
            i += 1
            continue

        if current_section in ("Args", "Parameters", "Attributes", "Variables"):
            param_match = re.match(
                r"^\s{4,}(\*?\*?\w+)\s*(?:\(([^)]+)\))?\s*(?:–|-|:)\s*(.*)", line
            )
            if param_match:
                current_items.append(
                    {
                        "name": param_match.group(1),
                        "type": param_match.group(2) or "",
                        "desc": param_match.group(3).strip(),
                    }
                )
                i += 1
                while i < len(lines) and lines[i].startswith("        "):
                    current_items[-1]["desc"] += " " + lines[i].strip()
                    i += 1
                continue

        bullet_match = re.match(r"^\s*[-*]\s+(.+)", stripped)
        if bullet_match and current_section:
            current_items.append({"text": bullet_match.group(1)})
            i += 1
            while (
                i < len(lines)
                and lines[i].startswith("  ")
                and not lines[i].strip().startswith("-")
            ):
                current_items[-1]["text"] += " " + lines[i].strip()
                i += 1
            continue

        if current_section and stripped:
            current_text.append(stripped)
        elif not current_section and stripped:
            if "description" not in result:
                result["description"] = []
            result["description"] = result.get("description", [])
            result["description"].append(stripped)

        i += 1

    if current_section:
        result["sections"].append(
            {
                "type": current_section,
                "items": current_items,
                "text": "\n".join(current_text).strip(),
            }
        )

    return result


def _render_docstring_html(doc: str, indent: str = "                ") -> str:
    """Render a docstring as structured HTML with proper sections."""
    if not doc:
        return ""

    parsed = _parse_google_docstring(doc)
    parts = []

    if parsed["summary"]:
        summary = _inline_format(html.escape(parsed["summary"]))
        parts.append(f"{indent}<p>{summary}</p>")

    for desc_line in parsed.get("description", []):
        parts.append(f"{indent}<p>{_inline_format(html.escape(desc_line))}</p>")

    for section in parsed.get("sections", []):
        sec_type = section["type"]
        items = section.get("items", [])
        text = section.get("text", "")

        if sec_type in ("Args", "Parameters"):
            parts.append(f'{indent}<div class="params-section">')
            parts.append(f"{indent}    <h5>Parameters</h5>")
            parts.append(f"{indent}    <dl>")
            for item in items:
                name = html.escape(item["name"])
                typ = html.escape(item.get("type", ""))
                desc = _inline_format(html.escape(item.get("desc", "")))
                type_html = (
                    f' <span class="param-type">({typ})</span>' if typ else ""
                )
                parts.append(
                    f"{indent}        <dt><strong>{name}</strong>{type_html}</dt>"
                )
                parts.append(f"{indent}        <dd>{desc}</dd>")
            parts.append(f"{indent}    </dl>")
            parts.append(f"{indent}</div>")

        elif sec_type in ("Returns", "Return type"):
            parts.append(f'{indent}<div class="returns-section">')
            parts.append(f"{indent}    <h5>Returns</h5>")
            if text:
                parts.append(
                    f"{indent}    <p>{_inline_format(html.escape(text))}</p>"
                )
            for item in items:
                parts.append(
                    f"{indent}    <p>{_inline_format(html.escape(item.get('text', '')))}</p>"
                )
            parts.append(f"{indent}</div>")

        elif sec_type in ("Note", "Warning"):
            css_class = "note" if sec_type == "Note" else "warning"
            parts.append(f'{indent}<div class="callout callout-{css_class}">')
            parts.append(f"{indent}    <strong>{sec_type}</strong>")
            if text:
                parts.append(
                    f"{indent}    <p>{_inline_format(html.escape(text))}</p>"
                )
            parts.append(f"{indent}</div>")

        elif sec_type in ("Example", "Models"):
            parts.append(f'{indent}<div class="example-section">')
            parts.append(f"{indent}    <h5>{sec_type}</h5>")
            if text:
                parts.append(
                    f"{indent}    <p>{_inline_format(html.escape(text))}</p>"
                )
            if items:
                parts.append(f"{indent}    <ul>")
                for item in items:
                    item_text = _inline_format(html.escape(item.get("text", "")))
                    parts.append(f"{indent}        <li>{item_text}</li>")
                parts.append(f"{indent}    </ul>")
            parts.append(f"{indent}</div>")

        elif sec_type == "Raises":
            parts.append(f'{indent}<div class="raises-section">')
            parts.append(f"{indent}    <h5>Raises</h5>")
            parts.append(f"{indent}    <dl>")
            for item in items:
                name = html.escape(item.get("name", ""))
                desc = _inline_format(html.escape(item.get("desc", "")))
                parts.append(f"{indent}        <dt><code>{name}</code></dt>")
                parts.append(f"{indent}        <dd>{desc}</dd>")
            parts.append(f"{indent}    </dl>")
            parts.append(f"{indent}</div>")

        else:
            if text:
                parts.append(
                    f"{indent}<p><strong>{html.escape(sec_type)}:</strong> "
                    f"{_inline_format(html.escape(text))}</p>"
                )
            if items:
                parts.append(f"{indent}<ul>")
                for item in items:
                    parts.append(
                        f"{indent}    <li>"
                        f"{_inline_format(html.escape(item.get('text', '')))}</li>"
                    )
                parts.append(f"{indent}</ul>")

    return "\n".join(parts)


def _inline_format(text: str) -> str:
    """Apply inline formatting: backtick code, bold, etc."""
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    return text


# ---------------------------------------------------------------------------
# HTML Rendering
# ---------------------------------------------------------------------------

NAV_HTML = """\
    <nav class="docs-nav">
        <div class="nav-inner">
        <a href="/quakfilter/" class="nav-brand">
            <img src="../../favicon.svg" alt="Vistralis">
            Vistralis <span class="nav-breadcrumb">/</span> quakfilter
            <span class="nav-breadcrumb">/</span> API
        </a>
        <ul class="nav-links">
            <li><a href="/quakfilter/">Overview</a></li>
            <li><a href="./" class="active">API Docs</a></li>
            <li><a href="https://github.com/vistralis/quakfilter"
                   target="_blank" rel="noopener noreferrer">GitHub</a></li>
            <li><a href="https://pypi.org/project/quakfilter/"
                   target="_blank" rel="noopener noreferrer">PyPI</a></li>
        </ul>
        </div>
    </nav>
"""


def render_sidebar(active_slug: str, module_data: dict[str, Any]) -> str:
    """Render sidebar with page links and per-page class anchors."""
    lines = ['    <aside class="docs-sidebar">']
    lines.append('        <div class="sidebar-section">')
    lines.append("            <h4>Getting Started</h4>")

    active_cls = ' class="active"' if active_slug == "index" else ""
    lines.append(f'            <a href="index.html"{active_cls}>Overview</a>')
    lines.append("        </div>")

    lines.append('        <div class="sidebar-section">')
    lines.append("            <h4>API Reference</h4>")
    for page_def in PAGE_DEFINITIONS:
        slug = page_def["slug"]
        title = page_def["title"]
        is_active = active_slug == slug
        active_cls = ' class="active"' if is_active else ""
        lines.append(f'            <a href="{slug}.html"{active_cls}>{title}</a>')

        if is_active:
            for class_name in page_def["classes"]:
                if class_name in module_data["classes"]:
                    lines.append(
                        f'            <a href="#{class_name}" '
                        f'class="sidebar-sub">{class_name}</a>'
                    )
    lines.append("        </div>")
    lines.append("    </aside>")
    return "\n".join(lines)


def render_method_card(method: dict[str, Any], class_name: str) -> str:
    """Render a single method as a structured card."""
    name = method["name"]
    sig = html.escape(method["signature"])

    badge = ""
    if method.get("is_classmethod") or name.startswith("from_"):
        badge = '<span class="badge badge-classmethod">classmethod</span> '
    elif method.get("is_staticmethod"):
        badge = '<span class="badge badge-static">static</span> '

    anchor_id = f"{class_name}.{name}"

    lines = [f'            <div class="method-card" id="{anchor_id}">']
    lines.append(
        f'                <div class="method-header">'
        f'{badge}<code class="method-name">{class_name}.{name}</code>'
        f"</div>"
    )
    lines.append(
        f'                <pre class="signature"><code>'
        f"{html.escape(name)}{sig}</code></pre>"
    )

    doc_html = _render_docstring_html(method["docstring"])
    if doc_html:
        lines.append(doc_html)

    lines.append("            </div>")
    return "\n".join(lines)


def render_enum_section(cls_data: dict[str, Any]) -> str:
    """Render an Enum class as a variants table."""
    name = cls_data["name"]
    doc = cls_data["docstring"]

    lines = [f'        <div class="class-section" id="{name}">']
    lines.append(
        f'            <h3><code class="class-name">class {name}</code>'
        f' <span class="class-tag">Enum</span></h3>'
    )

    doc_html = _render_docstring_html(doc, indent="            ")
    if doc_html:
        lines.append(doc_html)

    enum_props = [
        p
        for p in cls_data["properties"]
        if p["name"] not in ("name", "value") and not p["name"].startswith("_")
    ]
    if enum_props:
        lines.append('            <div class="params-section">')
        lines.append("                <h5>Members</h5>")
        lines.append("                <table>")
        lines.append("                    <tr><th>Name</th><th>Value</th></tr>")
        for prop in enum_props:
            pname = html.escape(prop["name"])
            lines.append(
                f"                    <tr>"
                f"<td><code>{pname}</code></td>"
                f"<td>{html.escape(prop['docstring'] or '—')}</td></tr>"
            )
        lines.append("                </table>")
        lines.append("            </div>")

    own_methods = [m for m in cls_data["methods"] if m.get("is_own", True)]
    if own_methods:
        lines.append("            <h4>Methods</h4>")
        for method in own_methods:
            lines.append(render_method_card(method, name))

    lines.append("        </div>")
    return "\n".join(lines)


def render_class_section(cls_data: dict[str, Any]) -> str:
    """Render a class section with proper structure."""
    if cls_data.get("is_enum"):
        return render_enum_section(cls_data)

    name = cls_data["name"]
    doc = cls_data["docstring"]
    bases = cls_data["bases"]
    is_thin = cls_data.get("is_thin", False)

    lines = [f'        <div class="class-section" id="{name}">']

    base_str = ""
    if bases:
        base_names = [b.split(".")[-1] for b in bases if "ABC" not in b]
        if base_names:
            base_str = f'({", ".join(base_names)})'

    tag = ""
    if any("ABC" in b for b in bases):
        tag = ' <span class="class-tag">abstract</span>'
    elif is_thin:
        tag = f' <span class="class-tag">extends {base_str.strip("()")}</span>'

    lines.append(
        f'            <h3><code class="class-name">class {name}</code>'
        f"{base_str}{tag}</h3>"
    )

    doc_html = _render_docstring_html(doc, indent="            ")
    if doc_html:
        lines.append(doc_html)

    props = [
        p
        for p in cls_data["properties"]
        if not p["name"].startswith("_") and p["docstring"]
    ]
    if props:
        lines.append('            <div class="params-section">')
        lines.append("                <h5>Properties</h5>")
        lines.append("                <dl>")
        for prop in props:
            pname = html.escape(prop["name"])
            pdoc_text = _inline_format(html.escape(prop["docstring"]))
            lines.append(f"                    <dt><code>{pname}</code></dt>")
            lines.append(f"                    <dd>{pdoc_text}</dd>")
        lines.append("                </dl>")
        lines.append("            </div>")

    constructors = [
        m
        for m in cls_data["methods"]
        if (m["name"].startswith("from_") or m["name"] == "__init__")
        and m.get("is_own", True)
    ]
    instance_methods = [
        m
        for m in cls_data["methods"]
        if m not in constructors
        and not m["name"].startswith("_")
        and m.get("is_own", True)
    ]

    if constructors:
        lines.append('            <h4 class="section-label">Constructors</h4>')
        for method in constructors:
            lines.append(render_method_card(method, name))

    if instance_methods:
        lines.append('            <h4 class="section-label">Methods</h4>')
        for method in instance_methods:
            lines.append(render_method_card(method, name))

    lines.append("        </div>")
    return "\n".join(lines)


def render_function_section(func_data: dict[str, Any]) -> str:
    """Render a module-level function."""
    name = func_data["name"]
    sig = html.escape(func_data["signature"])

    lines = [f'        <div class="method-card" id="{name}">']
    lines.append(
        f'            <div class="method-header">'
        f'<code class="method-name">{name}</code></div>'
    )
    lines.append(
        f'            <pre class="signature"><code>'
        f"{html.escape(name)}{sig}</code></pre>"
    )

    doc_html = _render_docstring_html(func_data["docstring"])
    if doc_html:
        lines.append(doc_html)

    lines.append("        </div>")
    return "\n".join(lines)


def render_page(
    page_def: dict[str, Any],
    module_data: dict[str, Any],
) -> str:
    """Render a full page."""
    slug = page_def["slug"]
    title = page_def["title"]
    subtitle = page_def["subtitle"]
    version = module_data["version"]

    content_parts = []

    for class_name in page_def["classes"]:
        if class_name in module_data["classes"]:
            content_parts.append(
                render_class_section(module_data["classes"][class_name])
            )

    func_names = page_def.get("functions", [])
    if func_names:
        rendered_funcs = []
        for func_name in func_names:
            if func_name in module_data["functions"]:
                rendered_funcs.append(
                    render_function_section(module_data["functions"][func_name])
                )
        if rendered_funcs:
            content_parts.append('        <h2 id="functions">Module Functions</h2>')
            content_parts.extend(rendered_funcs)

    content_html = "\n\n".join(content_parts)

    return PAGE_TEMPLATE.format(
        title=title,
        subtitle=subtitle,
        version=version,
        nav=NAV_HTML,
        sidebar=render_sidebar(slug, module_data),
        content=content_html,
    )


def render_index(module_data: dict[str, Any]) -> str:
    """Render the index page."""
    version = module_data["version"]

    total_classes = sum(len(p["classes"]) for p in PAGE_DEFINITIONS)
    total_funcs = sum(len(p.get("functions", [])) for p in PAGE_DEFINITIONS)

    cards_html = []
    for page_def in PAGE_DEFINITIONS:
        slug = page_def["slug"]
        title = page_def["title"]
        subtitle = page_def["subtitle"]
        num_classes = len(page_def["classes"])
        num_funcs = len(page_def.get("functions", []))
        count_parts = []
        if num_classes:
            count_parts.append(
                f"{num_classes} class{'es' if num_classes > 1 else ''}"
            )
        if num_funcs:
            count_parts.append(
                f"{num_funcs} function{'s' if num_funcs > 1 else ''}"
            )
        count_str = ", ".join(count_parts)

        cards_html.append(
            f'            <a href="{slug}.html" class="docs-card">\n'
            f"                <h3>{title}</h3>\n"
            f"                <p>{subtitle}</p>\n"
            f'                <span class="card-count">{count_str}</span>\n'
            f"            </a>"
        )

    return INDEX_TEMPLATE.format(
        version=version,
        nav=NAV_HTML,
        sidebar=render_sidebar("index", module_data),
        cards_grid="\n".join(cards_html),
        total_classes=total_classes,
        total_funcs=total_funcs,
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

PAGE_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — quakfilter API</title>
    <meta name="description"
        content="{subtitle} — quakfilter v{version} API reference.">
    <link rel="icon" type="image/svg+xml" href="../../favicon.svg">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
        rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css">
    <link rel="stylesheet" href="styles.css">
</head>

<body>
{nav}

    <div class="docs-wrapper">
{sidebar}

    <main class="docs-content">
        <h1>{title}</h1>
        <p class="subtitle">{subtitle}</p>

{content}
    </main>
    </div>

    <footer class="docs-footer">
        <span>&copy; 2026 Vistralis. All rights reserved.</span>
        <span>quakfilter v{version}</span>
    </footer>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>

</html>
"""

INDEX_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>API Reference — quakfilter</title>
    <meta name="description"
        content="API reference for quakfilter — modular state estimation
        building blocks for robotics and computer vision.">
    <link rel="icon" type="image/svg+xml" href="../../favicon.svg">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap"
        rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/styles/github-dark.min.css">
    <link rel="stylesheet" href="styles.css">
</head>

<body>
{nav}

    <div class="docs-wrapper">
{sidebar}

    <main class="docs-content">
        <h1>API Reference</h1>
        <p class="subtitle">
            quakfilter v{version} — {total_classes} classes,
            {total_funcs} functions
        </p>

        <p>quakfilter provides modular state estimation building blocks —
        Unscented Kalman Filters, Interacting Multiple Model estimators,
        and a multi-object track pool — all backed by a high-performance
        C++ core with full Python bindings.</p>

        <div class="callout callout-note">
            <strong>Install</strong>
            <p><code>pip install quakfilter</code></p>
        </div>

        <h2>API Sections</h2>

        <div class="cards-grid">
{cards_grid}
        </div>

        <h2>Quick Start</h2>

        <pre><code class="language-python">import numpy as np
import quak

# Constant-velocity rigid body model
model = quak.RigidBodyCVModel()

# 16D state: position(3) + velocity(3) + quaternion(4) + angular velocity(3×2)
x0 = np.zeros(16)
x0[9] = 1.0  # quaternion w-component

ukf = quak.UKF(
    model=model,
    process_noise=np.eye(15) * 0.01,
    measurement_noise=np.eye(6) * 0.001,
    initial_covariance=np.eye(15) * 0.1,
    initial_state=x0,
)

# Predict-update loop
ukf.predict(dt=0.1)
state = ukf.update(measurement)</code></pre>
    </main>
    </div>

    <footer class="docs-footer">
        <span>&copy; 2026 Vistralis. All rights reserved.</span>
        <span>quakfilter v{version}</span>
    </footer>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.11.1/highlight.min.js"></script>
    <script>hljs.highlightAll();</script>
</body>

</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Build API documentation."""
    parser = argparse.ArgumentParser(
        description="Build quakfilter API docs as static HTML."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: website/quakfilter/api/)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    if args.output:
        output_dir = args.output.resolve()
    else:
        output_dir = project_root / "website" / "quakfilter" / "api"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting API structure from quak...")
    module_data = extract_module()
    print(
        f"  Found {len(module_data['classes'])} classes, "
        f"{len(module_data['functions'])} functions "
        f"(v{module_data['version']})"
    )

    # Copy styles.css
    styles_src = Path(__file__).parent / "api_styles.css"
    styles_dst = output_dir / "styles.css"
    if styles_src.exists():
        styles_dst.write_text(styles_src.read_text())
        print("  Copied styles.css")

    # Render index
    index_html = render_index(module_data)
    (output_dir / "index.html").write_text(index_html)
    print("  Wrote index.html")

    # Render each page
    for page_def in PAGE_DEFINITIONS:
        page_html = render_page(page_def, module_data)
        filename = f"{page_def['slug']}.html"
        (output_dir / filename).write_text(page_html)
        print(f"  Wrote {filename}")

    print(f"\nDocs built → {output_dir}")


if __name__ == "__main__":
    main()
