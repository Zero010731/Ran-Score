from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from openai import OpenAI


LABEL_COLUMNS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Lung Opacity",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
    "emphysema",
    "interstitial lung disease",
    "calcification(lung and mediastinal)",
    "Trachea and bronchus",
    "cavity and cyst",
    "mediastinal other",
    "pulmonary vascular abnormal",
]


prompt_templates = {
    "Atelectasis": 
    """
    Your task is to identify and label the presence of Atelectasis in the given text reports.
    Look for phrases in the report that describe the lung fields. If the report does not mention terms like "homogeneous opacity" or "increased density" in a specific lobe or segment, it may suggest that Atelectasis is not present. Note the absence of such terms and any other signs of Atelectasis in the lung fields.
    Check if the report mentions the presence or absence of air-bronchograms within the affected area. If air-bronchograms are not mentioned, it may indicate that Atelectasis is not present or that the atelectasis is complete. The absence of air-bronchograms can be a key indicator.
    Evaluate the description of pulmonary vessels and bronchial markings in the report. If the report does not mention that the vessels appear crowded or that the bronchial markings are displaced or distorted, it may suggest that Atelectasis is not present. Note the absence of these changes in lung markings.
    Observe the report's description of the diaphragm contour and costophrenic angles. If the report does not mention any abnormalities such as an elevated diaphragm on the affected side or blunting of the costophrenic angle, it may indicate that Atelectasis is not present. Note the absence of these abnormalities.
    Note the report's comments on the heart size and shape, as well as any shifts in the mediastinum. If the report does not mention any mediastinal shift or other abnormalities related to the heart and mediastinum that are consistent with Atelectasis, it may suggest that Atelectasis is not present. Note the absence of these abnormalities.
    """,
    "Cardiomegaly":
    """
    Your task is to identify and label the presence of Cardiomegaly in the given text reports.
    Look for phrases in the report that describe the heart size. 
    Terms like "enlarged heart," "cardiomegaly," "borderline enlarged," "mildly enlarged," or "somewhat enlarged" should be interpreted as indicating cardiomegaly, unless the report clearly attributes the appearance to a non-cardiac cause (e.g., low lung volumes or technical factors).
    Example:
    1.If the report states "cardiac size is enlarged / mildly enlarged / borderline enlarged / somewhat enlarged," label as Cardiomegaly.
    2.If the report states "prominent heart due to low lung volumes" or other artifact explanations, do not label as Cardiomegaly.
    """ ,
    "Consolidation": 
    """
    Your task is to identify and label the presence of Consolidation in the given text reports.
    Look specifically for phrases that clearly describe "consolidation" or "lung parenchymal" "airspace disease","alveolar consolidation","lobar consolidation","focal consolidation."
    Terms like "consolidation," "airspace consolidation," "lobar consolidation," or "dense airspace opacity" should be interpreted as indicating consolidation.
    Example:
    1.If the report states "There is consolidation in the right lower lobe" -> label as Consolidation.
    2.If the report states "Diffuse bilateral opacities, likely pulmonary edema" -> do NOT label as Consolidation.
    3.If the report states "Opacity could represent pneumonia, cannot exclude consolidation" -> do NOT label as Consolidation.
    """,
    "Edema": 
    """
    Your task is to identify and label the presence of Edema in the given text reports.
    Look specifically for any mention of pulmonary edema.
    Terms like "pulmonary edema," "minimal edema," "resolved edema," or "edema or aspiration" should all be interpreted as indicating edema.
    """,
    "Enlarged Cardiomediastinum": 
    """
    Your task is to identify and label the presence of Enlarged Cardiomediastinum in the given text reports.
    Look specifically for phrases that clearly describe "cardiomegaly," "enlarged heart," "borderline enlarged cardiac silhouette," "enlarged cardiomediastinum," or "mediastinal widening."
    Terms like "mild cardiomegaly," "top-normal to mildly enlarged heart," "borderline enlarged heart," or "unchanged cardiomegaly" should be interpreted as indicating Enlarged Cardiomediastinum.
    Example:
    1. If the report states "Borderline cardiomegaly is present" -> label as Enlarged Cardiomediastinum.
    2. If the report states "The mediastinal contours are normal" -> do NOT label as Enlarged Cardiomediastinum.
    3. If the report states "Mild cardiomegaly, unchanged from prior study" -> label as Enlarged Cardiomediastinum.
    """,
    "Fracture": 
    """
    Your task is to identify and label the presence of fracture in the given text reports.
    Look specifically for phrases that describe any bone fractures, including rib fractures, vertebral compression fractures, healed fractures, or deformities of the spine.
    Terms like "healed fracture," "old fracture," "rib fracture," "compression deformity," "vertebral compression fracture," or "stable from prior imaging" should all be interpreted as indicating fracture.
    """,
    "Lung Lesion":  
    """
    Your task is to identify and label the presence of Lung Lesion in the given text reports.
    Look specifically for phrases that clearly describe structural lung abnormalities, such as nodules, masses, cavities, fibrosis, or bronchiectasis.
    Terms indicating non-structural, functional, or indeterminate findings should NOT be labeled as Lung Lesion, including pneumonia, consolidation, atelectasis, effusion, pulmonary edema, or chronic/stable findings.
    Example:
    1. If the report states "Pulmonary nodule is seen in the right upper lobe" -> label as Lung Lesion.
    2. If the report states "New opacity, cannot exclude underlying mass" -> label as Lung Lesion.
    """,
    # "No Finding": None,  # The prompt for 'No Finding' can be handled separately.
    "Pleural Effusion": 
    """
    Your task is to identify and label the presence of Pleural Effusion in the given text reports.
    Look specifically for phrases that clearly describe fluid in the pleural space, including pleural effusion, small effusion, trace effusion, possible effusion, or fluid in the costophrenic angles, fissures, or basal regions.
    Terms indicating no effusion or unrelated findings should NOT be labeled as Pleural Effusion, including chest tubes, post-procedure changes, pneumothorax, stable/unchanged effusions, or other non-pleural fluid findings.
    Even if the effusion is improving or partially resolved, it should still be labeled as Pleural Effusion.
    Example:
    1. "Slight blunting of the bilateral costophrenic angles which may be due to very trace pleural effusions" -> label as Pleural Effusion.
    2. "Small amount of fluid is seen in the right minor fissure" -> label as Pleural Effusion.

    """,
    "Pleural Other": 
    """
    Your task is to identify and label the presence of Pleural Other(NOT Pleural Effusion) in the given text reports.
    Look specifically for phrases that clearly describe pleural abnormalities beyond simple effusion,
    such as loculated pleural effusion, significant pleural thickening, or pleural abnormalities associated with chest tube or drainage interventions.
    Do NOT label as Pleural Other when the report only mentions the presence of a chest tube, a routine post-procedure state, or a stable/unchanged effusion.
    Mentions of minor, non-progressive, or non-specific findings should also NOT be labeled as Pleural Other,
    including small pleural effusion, mild pleural thickening, minimal or trace effusion, pleural effusion due to edema or infection, or stable post-procedure effusion
    Example:
    1. If the report states "Large left pleural effusion, possibly developing a loculated component" -> label as 0 (NOT Pleural Other).
    2. If the report states "Biapical pleural thickening is again seen" -> label as 0 (NOT Pleural Other).

    """,
    "Lung Opacity": 
    """
    Your task is to identify and label the presence of Lung Opacity in the given text reports.
    Look specifically for phrases that clearly describe areas of increased lung density or decreased aeration,such as opacity, opacities, infiltrate, airspace disease, or consolidation.
    Examples:
    1. If the report states "Streaky biapical and basilar opacities most compatible with scarring" -> label as 0 (NOT Lung Opacity)
    2. If the report states "Small left pleural effusion with adjacent atelectasis in retrocardiac lung" -> label as 0 (NOT Lung Opacity)
    3. If the report states "Mild chronic interstitial lung disease, stable or slightly improved from prior imaging" -> label as 0 (NOT Lung Opacity)
    4. If the report states "Linear or streaky opacities in lower lobes likely due to low lung volumes or technical factors" -> label as 0 (NOT Lung Opacity)  
    5. If the report states "Chronic consolidation in the right upper lobe has improved, and scarring is responsible for more rightward shift of the hilus" -> label as 0 (NOT Lung Opacity)

    """,
    "Pneumonia": 
    """
    Your task is to identify and label the presence of Pneumonia(NOT include resolved, cleared, or no longer present.) in the given chest radiograph text reports.
    Look specifically for phrases that clearly describe pneumonia or early pneumonia,
    such as "pneumonia," "early pneumonia," "cannot exclude pneumonia," or "superimposed pneumonia would have to be considered."
    Do NOT label as Pneumonia when the report attributes findings to other causes,
    including atelectasis, aspiration, pulmonary edema, post-procedure changes, or nonspecific/minimal opacities and effusions without mention of pneumonia.
    Also do NOT label as Pneumonia when findings are described as fully resolved or cleared,  or when only "infection" is mentioned without pneumonia.  
    Example:
    1. "Lower lobe opacity seen on lateral view, cannot exclude early pneumonia" -> label as Pneumonia
    2. "Small retrocardiac opacity most likely represents atelectasis, superimposed pneumonia would have to be considered" -> label as Pneumonia
    3. "Right lower lobe opacities worrisome for aspiration" -> label as NOT Pneumonia
    4. "Complete clearing of right upper lobe pneumonia" -> label as NOT Pneumonia

    """,
    "Pneumothorax": 
    """
    Your task is to identify and label the presence of Pneumothorax in the given text reports.
    Look specifically for phrases that clearly describe pneumothorax,
    such as direct mentions of pneumothorax, air in the pleural cavity, or lung collapse related to pneumothorax.
    Do NOT label as Pneumothorax when the report only mentions unrelated pleural or lung findings without evidence of air in the pleural space.
    Mentions of minor, non-progressive, or non-specific findings should also NOT be labeled as Pneumothorax,
    including mild hyperlucency without collapse, vague descriptions of lucency not linked to pleural air, or stable post-procedure changes without new pneumothorax.

    """,
    "Support Devices":  
    """
    Your task is to identify and label the presence of Support Devices(NOT include device removed) in the given text reports. 
    Look for explicit mentions of support devices currently in place, such as catheters, PICC lines, central lines, chest tubes, pacemakers, feeding tubes, drains, endotracheal tubes, tracheostomy tubes, surgical clips, fixation hardware, sternotomy wires, stents, or valve replacements.

    """,
    "emphysema": 
    """
    Your task is to identify and label the presence of emphysema in the given text reports.
    Look specifically for phrases that clearly describe findings consistent with emphysema,
    such as direct mention of "emphysema", "chronic obstructive pulmonary disease (COPD)", "hyperinflated lungs", 
    descriptions of increased lung volumes, "flattened diaphragms", "reduced vascular markings", "air trapping", or "bullae".

    """,
    "interstitial lung disease": 
    """
    Your task is to identify and label the presence of interstitial_lung_disease in the given text reports.
    Look specifically for phrases that clearly describe findings consistent with interstitial lung disease,
    such as 'Interstitial edema', 'Interstitial abnormality', 'Interstitial pulmonary edema', 
    'Chronic interstitial lung disease', 'Increased interstitial markings', 
    'Focal opacities in the lung bases' (which could be related to atelectasis or infection in the context of interstitial lung disease).
    """,
    "calcification(lung and mediastinal)": 
    """
    Your task is to identify and label the presence of calcification in the given text reports. 
    Check for mentions of calcification in the lungs or mediastinum. 
    For mediastinum, also include calcification of the aortic arch, coronary arteries, and cardiac valves in addition to lymph nodes or mediastinal soft tissues. 
    """,
    "Trachea and bronchus": 
    """
    Your task is to identify and label the presence of Trachea and bronchus in the given text reports. 
    Carefully review the following chest X-ray reports and determine if there are any findings related to tracheobronchial lesions. 
    Specifically, look for mentions of bronchiectasis, cavitation, or any other abnormalities involving the trachea or bronchus. Exclude findings related to endotracheal tube placement or other non-lesional findings. 
    Do NOT label as Trachea and bronchus when the report only mentions:
    Endotracheal tube, catheter, or stent placement/removal and their malposition
    Air bronchograms or other findings confined to lung parenchyma
    Pulmonary infections, pneumonia, or cavitation/bronchiectasis confined to lung tissue
            
    """,
    "cavity and cyst": 
    """
    Your task is to identify and label the presence of cavity and cyst in the given text reports. 
    Look specifically for phrases that clearly describe findings consistent with cavitation.
    """,
    "mediastinal other": 
    """
    Your task is to identify and label the presence of Mediastinal Other in the given text reports.
    Look specifically for phrases that clearly describe true pathological findings of the mediastinum itself, such as mediastinal widening (not attributed to patient rotation or technical factors), mediastinal mass, abnormal prominence or thickening, mediastinal compression or shift directly described as mediastinal abnormality, and pneumomediastinum or pneumopericardium.
    Example:
    1. "Neo esophagus is mildly distended with air and fluid" -> label as Mediastinal Other
    2. "Prominence of the mediastinum is probably postsurgical" -> label as Mediastinal Other
    3. "Apparent widening of the mediastinum may relate to the technique" -> label as Mediastinal Other
    4. "Indentation on the left side of the trachea may be due to an enlarged left thyroid" -> label as Mediastinal Other
    5. "Widening of the upper mediastinum could be due to vascular congestion alone" -> label as Mediastinal Other
    6. "Cardiomegaly and widened mediastinum are stable" -> label as Mediastinal Other
    7. "Opacification at the base of the right lung is due to large mediastinal fat collection, not an abnormality" -> label as NOT Mediastinal Other
    """,
    "pulmonary vascular abnormal": 
    """
    Your task is to identify and label the presence of pulmonary vascular abnormal in the given text reports.
    Look specifically for phrases that describe direct vascular findings (e.g., "pulmonary vascular engorgement", "enlarged pulmonary arteries") or indirect vascular findings linked to cardiac issues (e.g., "volume overload", "cardiac decompensation", "cardiogenic pulmonary edema"). Label only if there is a direct mention of vascular changes such as "vascular congestion has worsened", "increased pulmonary vascular markings", or "engorged pulmonary arteries". 
    Do NOT label when the report describes related findings without specifying vascular involvement, such as "fluid overload" or "atelectasis". 
    Do NOT label as pulmonary vascular abnormalities when the report uses synonyms or non-listed terms (e.g., "fluid overload", "vascular crowding", "increased interstitial markings") or mentions pulmonary edema without an explicit cardiac link (e.g., "pulmonary edema" alone without "cardiogenic" or "heart failure"). 
    Reports stating resolved or improved abnormalities or focusing on non-vascular issues (e.g., "atelectasis", "pleural effusion") should not be labeled as pulmonary vascular abnormalities.
    Examples:
    1. "Prominence of the azygos vein contour and central pulmonary vascularity is unchanged" -> label as Pulmonary Vascular Abnormality
    2. "Central nodular opacities, diffuse interstitial lung markings, and cardiomegaly likely represent cardiogenic pulmonary edema" -> label as Pulmonary Vascular Abnormality

    """

}


def build_prompt(report: str, label: str, guidance: str) -> str:
    return f"""
Medical Imaging Report Analysis Task:
[Target Condition] {label}
[Report Content] {report}
{guidance}
Please perform the following:
1. Consider all relevant observations and terminology.
2. Return 1 if the target condition is confirmed.
3. Return 0 if it is excluded, not mentioned, or low probability.

Response Format:
Single numeric character (0/1) with no punctuation or explanations.
""".strip()


def parse_binary_response(text: str) -> int:
    match = re.search(r"(1|0)", text.strip())
    return int(match.group(1)) if match else 0


def predict_report(client: OpenAI, model: str, report: str, *, temperature: float = 0.0, max_tokens: int = 50) -> tuple[dict[str, int], dict[str, str]]:
    labels = {key: 0 for key in prompt_templates}
    raw_outputs: dict[str, str] = {}

    for label, guidance in prompt_templates.items():
        prompt = build_prompt(report=report, label=label, guidance=guidance)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a radiologist analyzing medical reports."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=42,
        )
        output = response.choices[0].message.content.strip()
        raw_outputs[label] = output
        labels[label] = parse_binary_response(output)

    labels["No Finding"] = 1 if all(value == 0 for value in labels.values()) else 0
    return labels, raw_outputs


def split_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    status_path = Path(args.status_file) if args.status_file else output_path.with_suffix(".progress.txt")
    error_path = Path(args.error_log) if args.error_log else output_path.with_suffix(".errors.log")

    df = pd.read_csv(input_path, encoding=args.encoding)
    if args.report_column not in df.columns:
        raise SystemExit(f"Report column '{args.report_column}' not found in {input_path}. Available columns: {list(df.columns)}")

    id_columns = split_columns(args.id_columns)
    for column in id_columns:
        if column not in df.columns:
            raise SystemExit(f"ID column '{column}' not found in {input_path}.")

    start_index = args.start
    if args.resume and status_path.exists():
        try:
            start_index = max(start_index, int(status_path.read_text().strip()) + 1)
        except ValueError:
            pass

    client = OpenAI(base_url=args.base_url, api_key=args.api_key or os.environ.get(args.api_key_env, "dummy"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = id_columns + LABEL_COLUMNS
    if args.include_raw_outputs:
        fieldnames += [f"raw_{label}" for label in prompt_templates]

    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        stop = len(df) if args.limit is None else min(len(df), start_index + args.limit)
        for idx in range(start_index, stop):
            row = df.iloc[idx]
            report = str(row[args.report_column])
            try:
                labels, raw_outputs = predict_report(
                    client,
                    args.model,
                    report,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                output_row = {column: row[column] for column in id_columns}
                output_row.update(labels)
                if args.include_raw_outputs:
                    output_row.update({f"raw_{label}": raw_outputs[label] for label in prompt_templates})
                writer.writerow(output_row)
                file.flush()
                status_path.write_text(str(idx), encoding="utf-8")
                print(f"processed row {idx}")
            except Exception as exc:  # keep long batch jobs resumable
                with error_path.open("a", encoding="utf-8") as err:
                    err.write(f"Row {idx} error: {exc}\n")
                print(f"row {idx} failed: {exc}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen-based 21-label finding extraction for Ran Score.")
    parser.add_argument("--input", required=True, help="Input CSV containing radiology reports or generated reports.")
    parser.add_argument("--output", required=True, help="Output CSV with one binary column per finding label.")
    parser.add_argument("--report-column", default="report", help="Name of the input text/report column.")
    parser.add_argument("--id-columns", default="study_id", help="Comma-separated ID columns to copy to the output.")
    parser.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint, e.g. vLLM.")
    parser.add_argument("--model", default="Qwen/Qwen3-14B", help="Model name served by the endpoint.")
    parser.add_argument("--api-key", default=None, help="API key. For local vLLM this can usually be omitted.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable for API key fallback.")
    parser.add_argument("--encoding", default="utf-8", help="Input CSV encoding.")
    parser.add_argument("--start", type=int, default=0, help="Start row index.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows to process.")
    parser.add_argument("--resume", action="store_true", help="Resume from the status file if present.")
    parser.add_argument("--status-file", default=None, help="Progress file. Defaults to OUTPUT.progress.txt.")
    parser.add_argument("--error-log", default=None, help="Error log. Defaults to OUTPUT.errors.log.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--include-raw-outputs", action="store_true", help="Also store raw model outputs per label.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

