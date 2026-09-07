"""Standalone JSON, Markdown, and escaped HTML reports; no network assets."""

import html
import json


def _overview(data):
    """Return a compact table and interpretation paragraphs before detailed evidence."""
    if "results" in data:
        header = ["Design", "Status", "Train", "Validation", "Test", "Excluded", "Outside domain"]
        rows, notes = [], []
        for name, result in data["results"].items():
            s = result.get("summary", {})
            fraction = s.get("outside_training_domain_fraction")
            rows.append([name, result["status"], *[s.get(k, "—") for k in ("n_train", "n_validation", "n_test", "n_excluded")],
                         f"{fraction:.1%}" if fraction is not None else "not evaluated"])
            if "generalization" in s:
                notes.append(f"{name}: {s['generalization']['supports']} {s['generalization']['does_not_establish']}")
        return header, rows, notes
    if "train" in data:
        scope = data.get("diagnostics", {}).get("generalization", {})
        return ["Partition", "Rows"], [[p.title(), len(data.get(p, []))] for p in ("train", "validation", "test", "excluded")], list(scope.values())
    return ["Rows", "Errors", "Warnings", "Information"], [[data["n_rows"], *[sum(f["severity"] == s for f in data.get("findings", [])) for s in ("error", "warning", "info")]]], []


def render(result, format="json"):
    if format == "json":
        return result.to_json()
    data = result.to_dict()
    title = "ChemData Auditor report" if "n_rows" in data else "SciSplit comparison" if "results" in data else "SciSplit report"
    header, rows, notes = _overview(data)
    if format == "markdown":
        lines = [f"# {title}", "", "Row references are zero-based positions in the original input.", ""]
        def cell(value):
            return html.escape(str(value)).replace("|", "&#124;").replace("\n", " ").replace("`", "&#96;").replace("[", "&#91;")
        lines += ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
        lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
        lines += [""] + [cell(note) + "\n" for note in notes]
        for finding in data.get("findings", []):
            # Escape Markdown metacharacters and embedded HTML in untrusted cell/rule text.
            def safe(text):
                return html.escape(str(text)).replace("`", "&#96;").replace("*", "&#42;").replace("#", "&#35;").replace("[", "&#91;")
            lines += [f"## {safe(finding['severity'].upper())}: {safe(finding['code'])}", "",
                      safe(finding["message"]), "", f"Rows: {safe(finding['rows'])}", "",
                      f"Action: {safe(finding['suggestion'])}", ""]
        lines += ["## Complete evidence and reproducible configuration", "", "<pre>", html.escape(json.dumps(data, indent=2)), "</pre>", ""]
        return "\n".join(lines)
    if format == "html":
        table = "<table><thead><tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in header) + "</tr></thead><tbody>"
        table += "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"
        table += "".join(f"<p>{html.escape(str(note))}</p>" for note in notes)
        cards = []
        for f in data.get("findings", []):
            cards.append(f"<article><h2>{html.escape(f['severity'].upper())}: {html.escape(f['code'])}</h2>"
                         f"<p>{html.escape(f['message'])}</p><p><b>Rows:</b> {html.escape(str(f['rows']))}</p>"
                         f"<p><b>Action:</b> {html.escape(f['suggestion'])}</p></article>")
        return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                f"<title>{title}</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#182b34}}"
                "article{border-left:4px solid #297e78;padding:8px 20px;margin:24px 0;background:#f4f8f8}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f6;padding:20px}h2{font-size:18px}"
                "table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #ccd6d6}</style>"
                f"<h1>{title}</h1><p>Row references are zero-based positions in the original input.</p>{table}{''.join(cards)}"
                f"<h2>Complete evidence and reproducible configuration</h2><pre>{html.escape(json.dumps(data, indent=2))}</pre></html>\n")
    raise ValueError("Report format must be json, markdown, or html.")
