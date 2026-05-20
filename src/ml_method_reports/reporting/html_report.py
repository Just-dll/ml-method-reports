from __future__ import annotations

import base64
import mimetypes
from datetime import datetime
from html import escape
from pathlib import Path

from fpdf import FPDF

from ml_method_reports.reporting.models import ExperimentReport, ReportSection
from ml_method_reports.reporting.serialization import (
    format_report_value,
    sanitize_table_rows,
    to_report_value,
)
from ml_method_reports.reporting.types import ReportMetadata, ReportValue, TableRows


class HtmlReportGenerator:
    def render(
        self,
        report: ExperimentReport,
        embed_images: bool = False,
        base_dir: str | Path | None = None,
    ) -> str:
        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subtitle = f"<p class=\"subtitle\">{escape(report.subtitle)}</p>" if report.subtitle else ""
        metadata = to_report_value(report.metadata)
        metadata_rows = self._render_table(
            [{"Key": key, "Value": value} for key, value in metadata.items()]
        )
        resolved_base_dir = Path(base_dir) if base_dir is not None else getattr(self, "_base_dir", Path.cwd())
        sections = "\n".join(
            self._render_section(section, embed_images=embed_images, base_dir=resolved_base_dir)
            for section in report.sections
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(report.title)}</title>
  <style>
    :root {{
      --ink: #182033;
      --muted: #657084;
      --paper: #fbfcff;
      --panel: #ffffff;
      --line: #dbe3ef;
      --accent: #0f766e;
      --accent-soft: #d9f3ef;
      --code: #111827;
    }}
    * {{ box-sizing: border-box; }}
    html {{ overflow-x: hidden; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(15, 118, 110, 0.14), transparent 32rem),
        linear-gradient(135deg, #f7fafc 0%, #eef4f8 100%);
      font-family: Georgia, "Times New Roman", serif;
      line-height: 1.55;
    }}
    main {{
      max-width: 1080px;
      width: 100%;
      margin: 0 auto;
      padding: 42px 20px 56px;
      overflow-x: hidden;
    }}
    header {{
      padding: 34px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255, 255, 255, 0.86);
      box-shadow: 0 24px 70px rgba(24, 32, 51, 0.10);
    }}
    h1, h2 {{
      margin: 0;
      letter-spacing: -0.03em;
      line-height: 1.08;
    }}
    h1 {{ font-size: clamp(2.1rem, 5vw, 4rem); }}
    h2 {{ font-size: 1.55rem; }}
    .subtitle, .generated, .section-content {{ color: var(--muted); }}
    .generated {{ margin-top: 18px; font-size: 0.94rem; }}
    section {{
      margin-top: 24px;
      padding: 26px;
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 14px;
      table-layout: fixed;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    th {{
      background: var(--accent-soft);
      color: #0b4f4a;
      font-family: "Trebuchet MS", Verdana, sans-serif;
      font-size: 0.84rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    pre {{
      margin: 16px 0 0;
      padding: 16px;
      max-width: 100%;
      color: #e5edf7;
      background: var(--code);
      border-radius: 14px;
      overflow-x: auto;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    code {{
      font-family: "Cascadia Mono", Consolas, monospace;
      white-space: inherit;
      overflow-wrap: anywhere;
      word-break: break-word;
    }}
    img.report-image {{
      display: block;
      max-width: 100%;
      margin: 18px auto 6px;
      border: 1px solid var(--line);
      border-radius: 16px;
      box-shadow: 0 18px 42px rgba(24, 32, 51, 0.12);
    }}
    figure {{ margin: 0; }}
    figcaption {{
      color: var(--muted);
      font-size: 0.92rem;
      text-align: center;
      margin-top: 8px;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>{escape(report.title)}</h1>
      {subtitle}
      <p class="generated">Generated {generated_at}</p>
    </header>
    <section>
      <h2>Metadata</h2>
      {metadata_rows or '<p class="section-content">No metadata provided.</p>'}
    </section>
    {sections}
  </main>
</body>
</html>
"""

    def save(self, report: ExperimentReport, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._base_dir = path.parent
        path.write_text(self.render(report, base_dir=path.parent), encoding="utf-8")
        return path

    def _render_section(
        self,
        section: ReportSection,
        *,
        embed_images: bool,
        base_dir: Path,
    ) -> str:
        content = (
            f"<p class=\"section-content\">{escape(section.content)}</p>"
            if section.content
            else ""
        )
        image = self._render_image(section, embed_images=embed_images, base_dir=base_dir)
        table = self._render_table(sanitize_table_rows(section.table or []))
        code = f"<pre><code>{escape(section.code)}</code></pre>" if section.code else ""
        return f"""<section>
  <h2>{escape(section.title)}</h2>
  {content}
  {image}
  {table}
  {code}
</section>"""

    def _render_image(self, section: ReportSection, *, embed_images: bool, base_dir: Path) -> str:
        if section.image_path is None:
            return ""
        src = self._image_src(section.image_path, embed_images=embed_images, base_dir=base_dir)
        if src is None:
            missing = escape(str(section.image_path).replace("\\", "/"))
            return (
                "<p class=\"section-content\">"
                f"Image unavailable: {missing}"
                "</p>"
            )
        image_path = escape(src)
        caption = (
            f"<figcaption>{escape(section.image_caption)}</figcaption>"
            if section.image_caption
            else ""
        )
        return (
            "<figure>"
            f"<img class=\"report-image\" src=\"{image_path}\" alt=\"{escape(section.title)}\">"
            f"{caption}</figure>"
        )

    def _image_src(
        self,
        image_path: str | Path,
        *,
        embed_images: bool,
        base_dir: Path,
    ) -> str | None:
        if not embed_images:
            return str(image_path).replace("\\", "/")

        resolved = self._resolve_image_path(image_path, base_dir)
        if not resolved.exists() or not resolved.is_file():
            return None
        mime_type, _ = mimetypes.guess_type(resolved.name)
        mime_type = mime_type or "application/octet-stream"
        encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _resolve_image_path(self, image_path: str | Path, base_dir: Path) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        return base_dir / path

    def _render_table(self, rows: TableRows) -> str:
        if not rows:
            return ""

        columns = list(rows[0].keys())
        headers = "".join(f"<th>{escape(str(column))}</th>" for column in columns)
        body_rows = []
        for row in rows:
            cells = "".join(
                f"<td>{escape(self._format_value(row.get(column)))}</td>"
                for column in columns
            )
            body_rows.append(f"<tr>{cells}</tr>")
        return (
            f"<table><thead><tr>{headers}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody></table>"
        )

    def _format_value(self, value: ReportValue) -> str:
        return format_report_value(value)


class PdfExperimentReportGenerator:
    def save(self, report: ExperimentReport, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._base_dir = path.parent

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_margins(left=14, top=14, right=14)
        pdf.set_auto_page_break(auto=True, margin=14)
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=18)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(182, 9, self._safe(report.title))
        if report.subtitle:
            pdf.set_font("Helvetica", size=10)
            pdf.set_text_color(90, 99, 116)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(182, 6, self._safe(report.subtitle))
        pdf.set_text_color(34, 38, 47)
        pdf.set_font("Helvetica", size=9)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, 7, self._safe(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        pdf.ln(9)

        self._draw_section(
            pdf,
            ReportSection(title="Metadata", table=[to_report_value(report.metadata)]),
        )
        for section in report.sections:
            self._draw_section(pdf, section)

        pdf.output(str(path))
        return path

    def _draw_section(self, pdf: FPDF, section: ReportSection) -> None:
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.set_text_color(25, 33, 52)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 8, self._safe(section.title))
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(52, 59, 73)
        if section.content:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, self._safe(section.content))
            pdf.ln(1)
        if section.table:
            self._draw_simple_table(pdf, section.table)
        if section.image_path:
            self._draw_image(pdf, section.image_path, section.image_caption)
        if section.code:
            pdf.set_fill_color(244, 247, 252)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(182, 5, self._safe(section.code), border=1, fill=True)
        pdf.ln(4)

    def _draw_simple_table(self, pdf: FPDF, rows: TableRows) -> None:
        if not rows:
            return
        sanitized_rows, columns = self._prepare_pdf_rows(rows)
        if not columns:
            return
        column_widths = self._column_widths(columns)
        start_x = pdf.l_margin

        pdf.set_font("Helvetica", style="B", size=8)
        pdf.set_fill_color(232, 238, 248)
        pdf.set_x(start_x)
        for column, width in zip(columns, column_widths, strict=True):
            pdf.cell(width, 7, self._safe(str(column))[:20], border=1, fill=True)
        pdf.ln()

        pdf.set_font("Helvetica", size=6.5)
        for row in sanitized_rows:
            pdf.set_x(start_x)
            for column, width in zip(columns, column_widths, strict=True):
                value = row.get(column, "")
                pdf.cell(width, 7, self._safe(self._format_value(value))[:22], border=1)
            pdf.ln()

    def _prepare_pdf_rows(
        self,
        rows: TableRows,
    ) -> tuple[TableRows, list[str]]:
        sanitized_rows = sanitize_table_rows(rows, float_digits=4)
        columns = list(sanitized_rows[0].keys())
        if self._is_model_comparison_table(columns):
            pdf_rows = [
                {
                    "model": self._short_model_name(str(row.get("model", ""))),
                    "type": self._short_type_name(str(row.get("type", ""))),
                    "acc": row.get("accuracy", ""),
                    "err": row.get("error_rate", ""),
                    "f1": row.get("f1", ""),
                    "fit_ms": row.get("fit_ms", row.get("fit_time_ms", "")),
                    "pred_ms": row.get("predict_ms", row.get("predict_time_ms", "")),
                }
                for row in sanitized_rows
            ]
            return pdf_rows, ["model", "type", "acc", "err", "f1", "fit_ms", "pred_ms"]
        return sanitized_rows, columns

    def _is_model_comparison_table(self, columns: list[str]) -> bool:
        required = {"model", "type", "accuracy", "error_rate", "f1"}
        has_fit_time = "fit_ms" in columns or "fit_time_ms" in columns
        has_predict_time = "predict_ms" in columns or "predict_time_ms" in columns
        return required.issubset(set(columns)) and has_fit_time and has_predict_time

    def _short_model_name(self, model_name: str) -> str:
        names = {
            "EtalonClassifier": "Etalon",
            "LogisticRegression": "LogReg",
            "RandomForest": "RF",
        }
        return names.get(model_name, model_name)

    def _short_type_name(self, type_name: str) -> str:
        if type_name.startswith("custom"):
            return "custom"
        if type_name.startswith("sklearn"):
            return "sklearn"
        return type_name

    def _column_widths(self, columns: list[str]) -> list[float]:
        if columns == ["model", "type", "acc", "err", "f1", "fit_ms", "pred_ms"]:
            return [26, 24, 20, 20, 20, 36, 36]
        if columns == ["model", "type", "accuracy", "error_rate", "fit_ms", "predict_ms"]:
            return [42, 26, 27, 27, 30, 30]
        if columns == ["model", "type", "accuracy", "error_rate", "fit_time_ms", "predict_time_ms"]:
            return [42, 26, 27, 27, 30, 30]
        usable_width = 182
        return [usable_width / max(len(columns), 1)] * len(columns)

    def _draw_image(
        self,
        pdf: FPDF,
        image_path: str | Path,
        caption: str | None,
    ) -> None:
        path = self._resolve_image_path(image_path)
        if not path.exists():
            return
        image_width = 170
        image_height = 105
        if pdf.get_y() + image_height + 16 > pdf.page_break_trigger:
            pdf.add_page()
        x = pdf.l_margin + (182 - image_width) / 2
        pdf.image(str(path), x=x, y=pdf.get_y(), w=image_width)
        pdf.ln(image_height + 2)
        if caption:
            pdf.set_font("Helvetica", size=8)
            pdf.set_text_color(90, 99, 116)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(182, 5, self._safe(caption))

    def _resolve_image_path(self, image_path: str | Path) -> Path:
        path = Path(image_path)
        if path.is_absolute():
            return path
        base_dir = getattr(self, "_base_dir", Path.cwd())
        return Path(base_dir) / path

    def _format_value(self, value: ReportValue) -> str:
        return format_report_value(value)

    def _safe(self, value: ReportValue) -> str:
        return str(value).encode("latin-1", errors="replace").decode("latin-1")


def build_supervised_experiment_report(
    result: object,
    title: str = "Supervised Experiment Report",
) -> ExperimentReport:
    metadata = _as_dict(getattr(result, "run_metadata", {}))
    sections = [
        ReportSection(
            title="Dataset / Run Metadata",
            table=_mapping_to_rows(metadata),
        ),
        ReportSection(
            title="Leaderboard",
            table=_nested_mapping_to_rows(getattr(result, "leaderboard", {}), label_name="Model"),
        ),
        ReportSection(
            title="Best Model",
            content=str(getattr(result, "best_model", "N/A")),
        ),
    ]

    feature_importances = getattr(result, "feature_importances", None)
    if feature_importances:
        sections.append(
            ReportSection(
                title="Feature Importances",
                table=_mapping_to_rows(
                    feature_importances,
                    key_name="Feature",
                    value_name="Importance",
                ),
            )
        )

    confusion_matrix = getattr(result, "confusion_matrix", None)
    if confusion_matrix:
        sections.append(
            ReportSection(
                title="Confusion Matrix",
                table=[
                    {"Row": index, "Values": values}
                    for index, values in enumerate(confusion_matrix)
                ],
            )
        )

    classification_report = getattr(result, "classification_report", None)
    if classification_report:
        sections.append(
            ReportSection(
                title="Classification Report",
                table=_nested_mapping_to_rows(classification_report, label_name="Class"),
            )
        )

    # TODO: Map ROC/PR summaries when those values become part of the core result DTO.
    return ExperimentReport(
        title=title,
        subtitle="Static report generated from a supervised experiment result.",
        sections=sections,
        metadata=metadata,
    )


def _as_dict(value: object) -> dict[str, ReportValue]:
    if hasattr(value, "as_dict"):
        return dict(value.as_dict())
    if isinstance(value, dict):
        return dict(value)
    return {}


def _mapping_to_rows(
    mapping: ReportMetadata,
    key_name: str = "Key",
    value_name: str = "Value",
) -> TableRows:
    return [{key_name: key, value_name: value} for key, value in mapping.items()]


def _nested_mapping_to_rows(mapping: object, label_name: str) -> TableRows:
    if not isinstance(mapping, dict):
        return []

    rows: TableRows = []
    for key, values in mapping.items():
        if isinstance(values, dict):
            rows.append({label_name: key, **values})
        else:
            rows.append({label_name: key, "Value": values})
    return rows

