"""Standalone JSON, Markdown, and escaped HTML reports; no network assets."""

import html
import json


def render(result, format="json"):
    if format == "json":
        return result.to_json()
    data = result.to_dict()
    title = "ChemData Auditor report" if "findings" in data else "SciSplit report"
    if format == "markdown":
        lines = [f"# {title}", "", "Row references are zero-based positions in the original input.", ""]
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
        cards = []
        for f in data.get("findings", []):
            cards.append(f"<article><h2>{html.escape(f['severity'].upper())}: {html.escape(f['code'])}</h2>"
                         f"<p>{html.escape(f['message'])}</p><p><b>Rows:</b> {html.escape(str(f['rows']))}</p>"
                         f"<p><b>Action:</b> {html.escape(f['suggestion'])}</p></article>")
        return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
                f"<title>{title}</title><style>body{{font:16px system-ui;max-width:1000px;margin:40px auto;padding:0 24px;color:#182b34}}"
                "article{border-left:4px solid #297e78;padding:8px 20px;margin:24px 0;background:#f4f8f8}"
                "pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f4f6;padding:20px}h2{font-size:18px}</style>"
                f"<h1>{title}</h1><p>Row references are zero-based positions in the original input.</p>{''.join(cards)}"
                f"<h2>Complete evidence and reproducible configuration</h2><pre>{html.escape(json.dumps(data, indent=2))}</pre></html>\n")
    raise ValueError("Report format must be json, markdown, or html.")
