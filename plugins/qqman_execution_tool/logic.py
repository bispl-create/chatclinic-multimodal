from __future__ import annotations

import os
import subprocess
from pathlib import Path

from app.models import CmplotAssociationRequest, QqmanAssociationRequest, RPlotArtifact, RPlotRequest, RPlotResponse


RPLOT_OUTPUT_DIR = Path(
    os.getenv(
        "RPLOT_OUTPUT_DIR",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/outputs/rplots",
    )
)
LOCAL_RSCRIPT = Path(
    os.getenv(
        "LOCAL_RSCRIPT_BIN",
        "/Users/jongcye/Documents/Codex/.local/micromamba/envs/rgenomics/bin/Rscript",
    )
)
R_PLOT_SCRIPT = Path(
    os.getenv(
        "R_VCF_PLOT_SCRIPT",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/app/scripts/render_vcf_r_plots.R",
    )
)
CM_PLOT_SCRIPT = Path(
    os.getenv(
        "R_CMPLOT_SCRIPT",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/app/scripts/render_cmplot_association.R",
    )
)
QQMAN_PLOT_SCRIPT = Path(
    os.getenv(
        "R_QQMAN_SCRIPT",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/app/scripts/render_qqman_association.R",
    )
)
QQMAN_REPO_DIR = Path(
    os.getenv(
        "QQMAN_REPO_DIR",
        "/Users/jongcye/Documents/Codex/workspace/bioinformatics_vcf_evidence_mvp/third_party/qqman",
    )
)


from app.utils.parsing import safe_prefix as _safe_prefix_base


def _safe_prefix(prefix: str | None, source_path: str) -> str:
    return _safe_prefix_base(prefix, source_path, tool="rplots")


def _artifact_title_from_name(name: str) -> tuple[str, str]:
    lower = name.lower()
    if "qqman" in lower and "manhattan" in lower:
        return "qqman-manhattan", "qqman Manhattan"
    if "qqman" in lower and "qq" in lower:
        return "qqman-qq", "qqman QQ"
    if "density" in lower:
        return "cmplot-density", "CMplot Density"
    if "manhattan" in lower:
        return "cmplot-manhattan", "CMplot Manhattan"
    if "qq" in lower:
        return "cmplot-qq", "CMplot QQ"
    if "qual" in lower:
        return "qual-histogram", "QUAL Histogram"
    if "missing" in lower:
        return "sample-missingness", "Sample Missingness"
    if "variant_class" in lower:
        return "variant-class", "Variant Class"
    return "plot", name


def run_r_vcf_plots(request: RPlotRequest) -> RPlotResponse:
    input_path = Path(request.vcf_path)
    if not input_path.exists():
        raise FileNotFoundError(f"VCF not found: {request.vcf_path}")
    if not LOCAL_RSCRIPT.exists():
        raise FileNotFoundError(f"Local Rscript runtime not found: {LOCAL_RSCRIPT}")
    if not R_PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"R plotting script not found: {R_PLOT_SCRIPT}")

    RPLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.vcf_path)
    output_dir = RPLOT_OUTPUT_DIR / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.png"):
        stale_file.unlink(missing_ok=True)
    warnings_path = output_dir / "warnings.txt"
    warnings_path.unlink(missing_ok=True)

    cmd = [
        str(LOCAL_RSCRIPT),
        str(R_PLOT_SCRIPT),
        str(input_path),
        str(output_dir),
        prefix,
        str(request.density_bin_size),
        str(warnings_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    artifacts: list[RPlotArtifact] = []
    for image_path in sorted(output_dir.glob("*.png")):
        plot_type, title = _artifact_title_from_name(image_path.name)
        artifacts.append(
            RPlotArtifact(
                plot_type=plot_type,
                title=title,
                image_path=str(image_path),
                api_path=f"/api/v1/files?path={image_path}",
                note=f"Generated from {input_path.name} using vcfR and R plotting utilities.",
            )
        )

    warnings: list[str] = []
    if warnings_path.exists():
        warnings = [line.strip() for line in warnings_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return RPlotResponse(
        tool="r-vcfr-cmplot",
        input_path=str(input_path),
        output_dir=str(output_dir),
        command_preview=" ".join(cmd),
        artifacts=artifacts,
        warnings=warnings,
    )


def run_cmplot_association(request: CmplotAssociationRequest) -> RPlotResponse:
    input_path = Path(request.association_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Association table not found: {request.association_path}")
    if not LOCAL_RSCRIPT.exists():
        raise FileNotFoundError(f"Local Rscript runtime not found: {LOCAL_RSCRIPT}")
    if not CM_PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"CMplot association script not found: {CM_PLOT_SCRIPT}")

    RPLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.association_path)
    output_dir = RPLOT_OUTPUT_DIR / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.png"):
        stale_file.unlink(missing_ok=True)
    warnings_path = output_dir / "warnings.txt"
    warnings_path.unlink(missing_ok=True)

    cmd = [
        str(LOCAL_RSCRIPT),
        str(CM_PLOT_SCRIPT),
        str(input_path),
        str(output_dir),
        prefix,
        str(warnings_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    artifacts: list[RPlotArtifact] = []
    for image_path in sorted(output_dir.glob("*.png")):
        plot_type, title = _artifact_title_from_name(image_path.name)
        artifacts.append(
            RPlotArtifact(
                plot_type=plot_type,
                title=title,
                image_path=str(image_path),
                api_path=f"/api/v1/files?path={image_path}",
                note=f"Generated from {input_path.name} using CMplot GWAS-style association rendering.",
            )
        )

    warnings: list[str] = []
    if warnings_path.exists():
        warnings = [line.strip() for line in warnings_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return RPlotResponse(
        tool="r-cmplot-association",
        input_path=str(input_path),
        output_dir=str(output_dir),
        command_preview=" ".join(cmd),
        artifacts=artifacts,
        warnings=warnings,
    )


def run_qqman_association(request: QqmanAssociationRequest) -> RPlotResponse:
    input_path = Path(request.association_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Association table not found: {request.association_path}")
    if not LOCAL_RSCRIPT.exists():
        raise FileNotFoundError(f"Local Rscript runtime not found: {LOCAL_RSCRIPT}")
    if not QQMAN_PLOT_SCRIPT.exists():
        raise FileNotFoundError(f"qqman association script not found: {QQMAN_PLOT_SCRIPT}")
    if not QQMAN_REPO_DIR.exists():
        raise FileNotFoundError(f"qqman repository not found: {QQMAN_REPO_DIR}")

    RPLOT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = _safe_prefix(request.output_prefix, request.association_path)
    output_dir = RPLOT_OUTPUT_DIR / prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in output_dir.glob("*.png"):
        stale_file.unlink(missing_ok=True)
    warnings_path = output_dir / "warnings.txt"
    warnings_path.unlink(missing_ok=True)

    cmd = [
        str(LOCAL_RSCRIPT),
        str(QQMAN_PLOT_SCRIPT),
        str(input_path),
        str(output_dir),
        prefix,
        str(QQMAN_REPO_DIR),
        str(warnings_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)

    artifacts: list[RPlotArtifact] = []
    for image_path in sorted(output_dir.glob("*.png")):
        plot_type, title = _artifact_title_from_name(image_path.name)
        artifacts.append(
            RPlotArtifact(
                plot_type=plot_type,
                title=title,
                image_path=str(image_path),
                api_path=f"/api/v1/files?path={image_path}",
                note=f"Generated from {input_path.name} using the local qqman repository checkout.",
            )
        )

    warnings: list[str] = []
    if warnings_path.exists():
        warnings = [line.strip() for line in warnings_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return RPlotResponse(
        tool="r-qqman-association",
        input_path=str(input_path),
        output_dir=str(output_dir),
        command_preview=" ".join(cmd),
        artifacts=artifacts,
        warnings=warnings,
    )


def execute(payload: dict[str, object]) -> RPlotResponse:
    request = QqmanAssociationRequest(**payload)
    return run_qqman_association(request)
