"""Convert four-column CKG exports into the benchmark's three-column graphs.

The CKG exporter retains every relationship property in ``attributes_json``.
This mandatory stage applies the 31 source-specific preprocessing rules used
to obtain the relation vocabularies in the biomedical benchmark. Unsupported
CKG source files are skipped rather than passed to later construction stages
without their own preprocessing rule.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


CANONICAL_SOURCES = {
    "cgi": "CGI",
    "compartments": "COMPARTMENTS",
    "corum": "CORUM",
    "ctdhuman": "CTD_human",
    "dgidb": "DGIdb",
    "dip": "DIP",
    "diseases": "DISEASES",
    "disgenet": "DisGeNet",
    "drugbank": "DrugBank",
    "genomicsengland": "GENOMICS_ENGLAND",
    "panelapp": "GENOMICS_ENGLAND",
    "gwascatalog": "GWAS_Catalog",
    "hpidb": "HPIDb",
    "humanproteinatlaspathology": "Human_Protein_Atlas_pathology",
    "i2d": "I2D",
    "interologousinteractiondatabase": "I2D",
    "innatedb": "InnateDB",
    "intact": "IntAct",
    "intactmutationds": "Intact-MutationDs",
    "mutationds": "Intact-MutationDs",
    "mint": "MINT",
    "orphanet": "ORPHANET",
    "oncokb": "OncoKB",
    "pfam": "PFam",
    "psygenet": "PSYGENET",
    "reactome": "Reactome",
    "signor": "SIGNOR",
    "smpdb": "SMPDB",
    "stitch": "STITCH",
    "string": "STRING",
    "tissues": "TISSUES",
    "uniprot": "UniProt",
    "bhfucl": "bhf-ucl",
    "matrixdb": "matrixdb",
}


def source_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def canonical_source(stem: str) -> str | None:
    key = source_key(stem)
    return CANONICAL_SOURCES.get(key)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def interaction_relation(prefix: str, interaction: object, score: object) -> str:
    """Shared interaction hierarchy used by HPIDb-like source rules."""
    value = str(interaction).lower().strip() if interaction else ""
    if prefix == "HPI":
        if "phosphorylation" in value:
            base = "HPI_REACTION"
        elif "direct interaction" in value:
            base = "HPI_DIRECT_BINDING"
        elif "physical association" in value:
            base = "HPI_PHYSICAL_ASSOC"
        elif any(term in value for term in ("colocalization", "proximity", "association")):
            base = "HPI_SPATIAL_PROXIMITY"
        else:
            base = "HPI_GENERIC_INTERACT"
        return base + ("_HIGH" if as_float(score) >= 0.6 else "_LOW")

    if prefix == "I2D":
        if any(term in value for term in ("phospho", "ubiquitin", "sumoyl", "cleavage", "acetyl")):
            base = "I2D_BIOCHEMICAL"
        elif "direct interaction" in value:
            base = "I2D_DIRECT_BIND"
        elif "physical association" in value:
            base = "I2D_PHYSICAL_ASSOC"
        elif "proximity" in value or "colocalization" in value:
            base = "I2D_SPATIAL"
        else:
            base = "I2D_GENERAL"
        return base + ("_HIGH" if as_float(score) >= 0.7 else "_LOW")

    if prefix == "ECM":
        if any(term in value for term in ("enzymatic", "cleavage", "oxidoreductase")):
            base = "ECM_REMODELING_EVENT"
        elif "covalent" in value or "direct interaction" in value:
            base = "ECM_STRUCTURAL_BIND"
        elif "physical association" in value or "association" in value:
            base = "ECM_PHYSICAL_ASSOC"
        else:
            base = "ECM_SPATIAL_PROXIMITY"
        return base + ("_HIGH" if as_float(score) >= 0.6 else "_LOW")

    raise ValueError(f"Unknown shared interaction prefix: {prefix}")


def refine_relation(
    source: str,
    head: str,
    relation: str,
    attributes: dict[str, object],
) -> str:
    """Apply the rule associated with one canonical source name."""
    if source == "CGI" and relation == "CURATED_TARGETS":
        response = str(attributes.get("response", "")).lower().strip()
        if response == "responsive":
            return "DRUG_SENSITIVITY_IN"
        if response in {"resistant", "no responsive"}:
            return "DRUG_RESISTANCE_IN"
        if "increased toxicity" in response:
            return "DRUG_ADVERSE_REACTION"

    elif source == "COMPARTMENTS" and relation == "ASSOCIATED_WITH":
        if attributes.get("score") is None:
            return relation
        score = as_float(attributes.get("score"), float("nan"))
        if score != score:
            return relation
        if score >= 4.0:
            return "LOCATED_IN_COMPILED_HIGH"
        if score >= 2.0:
            return "LOCATED_IN_COMPILED_MEDIUM"
        return "LOCATED_IN_COMPILED_LOW"

    elif source == "CORUM" and relation == "IS_SUBUNIT_OF":
        evidence = attributes.get("evidences", [])
        evidence_text = (
            " ".join(map(str, evidence)).lower()
            if isinstance(evidence, list)
            else str(evidence).lower()
        )
        if any(term in evidence_text for term in ("x-ray", "crystallography", "electron microscopy", "nuclear magnetic resonance")):
            return "SUBUNIT_OF_STRUCTURAL_COMPLEX"
        if any(term in evidence_text for term in ("coimmunoprecipitation", "pull down", "chromatography", "purification")):
            return "SUBUNIT_OF_BIOCHEMICAL_COMPLEX"
        if any(term in evidence_text for term in ("text mining", "prediction", "inferred")):
            return "SUBUNIT_OF_PREDICTED_COMPLEX"
        return "SUBUNIT_OF_EXPERIMENTAL_COMPLEX"

    elif source == "CTD_human" and relation == "ASSOCIATED_WITH":
        if attributes.get("score") is None:
            return relation
        score = as_float(attributes.get("score"), float("nan"))
        if score != score:
            return relation
        if score == 1.0:
            return "ASSOCIATED_WITH_CORE_CONSENSUS"
        if score >= 0.8:
            return "ASSOCIATED_WITH_HIGH_CONFIDENCE"
        if score >= 0.6:
            return "ASSOCIATED_WITH_MODERATE_CONFIDENCE"
        return "ASSOCIATED_WITH_LOW_CONFIDENCE"

    elif source == "DGIdb" and relation == "CURATED_TARGETS":
        for value in (attributes.get("interaction_type", ""), attributes.get("score", "")):
            text = str(value).lower() if value else ""
            if any(term in text for term in ("inhibitor", "antagonist", "blocker", "negative", "inverse")):
                return "DRUG_INHIBITS_TARGET"
            if any(term in text for term in ("activator", "agonist", "inducer", "stimulator", "positive")):
                return "DRUG_ACTIVATES_TARGET"
            if "modulator" in text:
                return "DRUG_MODULATES_TARGET"
            if any(term in text for term in ("binder", "antibody", "cofactor")):
                return "DRUG_BINDS_TARGET"
        return "DRUG_INTERACTS_GENERIC"

    elif source == "DIP" and relation == "CURATED_INTERACTS_WITH":
        interaction = str(attributes.get("interaction_type", "")).lower()
        if "phospho" in interaction:
            base = "REACTION_PHOSPHORYLATION"
        elif "ubiquitin" in interaction or "neddyl" in interaction:
            base = "REACTION_UBIQUITINATION"
        elif "acetyl" in interaction:
            base = "REACTION_ACETYLATION"
        elif "cleavage" in interaction:
            base = "REACTION_CLEAVAGE"
        elif "methyl" in interaction:
            base = "REACTION_METHYLATION"
        elif "covalent" in interaction or "disulfide" in interaction:
            base = "BINDING_COVALENT"
        elif "direct interaction" in interaction:
            base = "BINDING_DIRECT"
        elif "physical association" in interaction:
            base = "BINDING_ASSOCIATION"
        elif "proximity" in interaction:
            base = "ASSOCIATION_PROXIMITY"
        else:
            base = "ASSOCIATION_GENERAL"
        return base + ("_HIGH" if as_float(attributes.get("score")) >= 0.7 else "_LOW")

    elif source == "DISEASES" and relation == "ASSOCIATED_WITH":
        if attributes.get("score") is None:
            return relation
        score = as_float(attributes.get("score"), float("nan"))
        if score != score:
            return relation
        if score >= 4.0:
            return "PROT_DISEASE_CONFIRMED"
        if score >= 3.0:
            return "PROT_DISEASE_HIGH_CONF"
        if score >= 1.5:
            return "PROT_DISEASE_MEDIUM_CONF"
        return "PROT_DISEASE_LOW_CONF"

    elif source == "DisGeNet" and relation == "ASSOCIATED_WITH":
        evidence = str(attributes.get("evidence_type", "")).lower().strip()
        score = as_float(attributes.get("score"))
        if evidence == "curated":
            if score == 1.0:
                return "GDA_CURATED_STRICT"
            if score >= 0.3:
                return "GDA_CURATED_SUPPORTIVE"
            return "GDA_CURATED_EMERGING"
        if evidence == "befree":
            return "GDA_NLP_VALIDATED" if score >= 0.1 else "GDA_NLP_POTENTIAL"
        return "GDA_UNKNOWN_ASSOCIATION"

    elif source == "DrugBank" and relation == "INTERACTS_WITH":
        text = str(attributes.get("interaction_type", "")).lower().strip()
        if re.search(r"risk|severity|bleeding|toxicity|adverse|prolong", text):
            return "DDI_RISK_ADVERSE"
        if re.search(r"increase|anticoagulant|enhance|elevated|higher", text):
            return "DDI_SYNERGY_INCREASE"
        if re.search(r"decrease|diminish|lessen|lower|reduced", text):
            return "DDI_ANTAGONISM_DECREASE"
        return "DDI_NEUTRAL_INTERACTION"

    elif source == "GENOMICS_ENGLAND" and relation == "ASSOCIATED_WITH":
        if attributes.get("score") is None:
            return relation
        score = as_float(attributes.get("score"), float("nan"))
        if score != score:
            return relation
        if score == 1.0:
            return "GENOMIC_DISEASE_CONFIRMED"
        if score >= 0.8:
            return "GENOMIC_DISEASE_STRONG"
        if score >= 0.6:
            return "GENOMIC_DISEASE_MODERATE"
        return "GENOMIC_DISEASE_SUPPORTIVE"

    elif source == "GWAS_Catalog":
        if relation == "STUDIES_TRAIT":
            return "STUDY_INVESTIGATES_PHENOTYPE"
        if relation == "VARIANT_FOUND_IN_GWAS":
            try:
                p_value = float(str(attributes.get("pvalue")).replace("E", "e"))
            except (ValueError, TypeError):
                return "VARIANT_ASSOCIATED_WITH_STUDY"
            base = "GWAS_SIGNIFICANT" if p_value <= 5e-8 else "GWAS_SUGGESTIVE" if p_value <= 1e-5 else "GWAS_NOMINAL"
            odds_raw = str(attributes.get("odds_ratio", "")).strip()
            if odds_raw not in {"NR", "0.0", "None", ""}:
                try:
                    odds = float(odds_raw)
                    if odds > 2.0 or odds < 0.5:
                        base += "_STRONG_EFFECT"
                except (ValueError, TypeError):
                    pass
            return base

    elif source == "HPIDb" and relation == "CURATED_INTERACTS_WITH":
        return interaction_relation("HPI", attributes.get("interaction_type"), attributes.get("score"))

    elif source == "Human_Protein_Atlas_pathology" and relation == "DETECTED_IN_PATHOLOGY_SAMPLE":
        try:
            positive = float(attributes.get("positive_prognosis_logrank_pvalue", 1.0))
            negative = float(attributes.get("negative_prognosis_logrank_pvalue", 1.0))
            if negative <= 0.05 and negative < positive:
                return "PROGNOSTIC_RISK_FACTOR"
            if positive <= 0.05 and positive < negative:
                return "PROGNOSTIC_FAVORABLE_FACTOR"
            high = int(attributes.get("expression_high", 0))
            medium = int(attributes.get("expression_medium", 0))
            return "CANCER_TISSUE_EXPRESSED" if high + medium > 0 else "CANCER_TISSUE_NOT_DETECTED"
        except (ValueError, TypeError):
            return relation

    elif source == "I2D" and relation == "CURATED_INTERACTS_WITH":
        return interaction_relation("I2D", attributes.get("interaction_type"), attributes.get("score"))

    elif source == "InnateDB" and relation == "CURATED_INTERACTS_WITH":
        interaction = str(attributes.get("interaction_type", "")).lower().strip()
        if any(term in interaction for term in ("phospho", "ubiquitin", "methyl", "neddyl")):
            base = "IMMUNE_MODULATION"
        elif "cleavage" in interaction:
            base = "IMMUNE_ACTIVATION_CLEAVAGE"
        elif "direct interaction" in interaction:
            base = "IMMUNE_PHYSICAL_DIRECT"
        elif "physical association" in interaction:
            base = "IMMUNE_PHYSICAL_ASSOC"
        else:
            base = "IMMUNE_SPATIAL_ASSOC"
        score = as_float(attributes.get("score"))
        return base + ("_GOLD" if score >= 0.8 else "_SILVER" if score >= 0.5 else "_BRONZE")

    elif source == "IntAct" and relation == "CURATED_INTERACTS_WITH":
        interaction = str(attributes.get("interaction_type", "")).lower().strip()
        if re.search(r"reaction|cleavage|ylation|methylation", interaction):
            base = "INTACT_REACTION"
        elif "direct interaction" in interaction or "covalent" in interaction:
            base = "INTACT_DIRECT_BIND"
        elif "physical association" in interaction:
            base = "INTACT_PHYSICAL_ASSOC"
        elif any(term in interaction for term in ("proximity", "colocalization", "association")):
            base = "INTACT_SPATIAL"
        else:
            base = "INTACT_GENERIC"
        score = as_float(attributes.get("score"))
        return base + ("_CORE" if score >= 0.7 else "_VETTED" if score >= 0.4 else "_CANDIDATE")

    elif source == "Intact-MutationDs" and relation == "CURATED_AFFECTS_INTERACTION_WITH":
        effect = str(attributes.get("effect", "")).lower().strip()
        if "disrupting" in effect:
            return "VAR_DISRUPTS_PPI"
        if "decreasing" in effect:
            return "VAR_DECREASES_PPI"
        if "increasing" in effect or "causing" in effect:
            return "VAR_INCREASES_PPI"
        if "no effect" in effect:
            return "VAR_NEUTRAL_PPI"
        return "VAR_AFFECTS_PPI_GENERIC"

    elif source == "MINT" and relation == "CURATED_INTERACTS_WITH":
        interaction = str(attributes.get("interaction_type", "")).lower().strip()
        if re.search(r"ribosylation|isomerization|neddylation|glycosylation", interaction):
            base = "MINT_COMPLEX_MOD"
        elif re.search(r"phosphorylation|ubiquitination|acetylation|cleavage|methylation", interaction):
            base = "MINT_PTM_EVENT"
        elif "direct interaction" in interaction or "covalent" in interaction:
            base = "MINT_DIRECT_BIND"
        elif "physical association" in interaction:
            base = "MINT_PHYSICAL_ASSOC"
        else:
            base = "MINT_SPATIAL_NEAR"
        score = as_float(attributes.get("score"))
        return base + ("_HIGH_CONF" if score >= 0.7 else "_MED_CONF" if score >= 0.4 else "_LOW_CONF")

    elif source == "ORPHANET" and relation == "ASSOCIATED_WITH":
        score = as_float(attributes.get("score"))
        if score >= 0.9:
            return "RD_CAUSATIVE_GENE"
        if score >= 0.7:
            return "RD_STRONG_ASSOCIATION"
        return "RD_GENE_ASSOCIATION"

    elif source == "OncoKB":
        evidence = str(attributes.get("evidence", "")).lower().strip()
        if relation == "TARGETS_CLINICALLY_RELEVANT_VARIANT":
            if evidence == "approved":
                return "DRUG_INDICATED_BY_VARIANT"
            if evidence == "r":
                return "DRUG_RESISTED_BY_VARIANT"
            if "clinical" in evidence:
                return "DRUG_CLINICAL_TRIAL_VARIANT"
            return "DRUG_BIOLOGICALLY_LINKED_VARIANT"
        if relation == "ASSOCIATED_WITH":
            try:
                publications = int(attributes.get("number_publications", 0))
            except (ValueError, TypeError):
                publications = 0
            return "VARIANT_CORE_DIAGNOSTIC_MARKER" if publications >= 5 else "VARIANT_PATHOGENIC_ASSOCIATION"
        if relation == "VARIANT_IS_CLINICALLY_RELEVANT":
            return "GENOME_TO_PROTEIN_VARIANT_MAP"

    elif source == "PFam" and relation == "FOUND_IN_PROTEIN":
        try:
            start = int(attributes.get("start", 0))
            domain_length = int(attributes.get("end", 0)) - start
        except (ValueError, TypeError):
            return relation
        if start <= 30:
            return "PFAM_N_TERMINAL_DOMAIN"
        if domain_length > 200:
            return "PFAM_LARGE_STRUCTURAL_BLOCK"
        return "PFAM_FUNCTIONAL_UNIT"

    elif source == "PSYGENET" and relation == "ASSOCIATED_WITH":
        score = as_float(attributes.get("score"))
        if score >= 0.75:
            return "PSY_GENETIC_CORE"
        if score >= 0.6:
            return "PSY_GENETIC_SUPPORTED"
        return "PSY_GENETIC_CANDIDATE"

    elif source == "Reactome" and relation == "ANNOTATED_IN_PATHWAY":
        base = "PW_CORE" if attributes.get("evidence", "IEA") == "TAS" else "PW_ASSOC"
        component = str(attributes.get("cellular_component", "")).lower().strip()
        if "membrane" in component:
            suffix = "_MEMB"
        elif "lumen" in component:
            suffix = "_LUME"
        elif "extracellular" in component or "exosome" in component:
            suffix = "_SEC"
        elif any(term in component for term in ("nuclear", "nucleo", "chromo")):
            suffix = "_NUCL"
        elif "cytosol" in component or "cytoplasm" in component:
            suffix = "_CYTO"
        else:
            suffix = "_GENERIC"
        return base + suffix

    elif source == "SIGNOR":
        if relation == "HAS_MODIFICATION":
            return "MOD_SPECIFIC_SITE"
        if relation == "IS_SUBSTRATE_OF":
            regulation = str(attributes.get("regulation", "")).lower().strip()
            if "up-regulates" in regulation:
                if "activity" in regulation:
                    return "SIG_ACT_STIMULATE"
                if "stabilization" in regulation:
                    return "SIG_STABILIZE"
                return "SIG_UPREGULATE"
            if "down-regulates" in regulation:
                if "activity" in regulation:
                    return "SIG_ACT_INHIBIT"
                if "destabilization" in regulation:
                    return "SIG_DEGRADE"
                return "SIG_DOWNREGULATE"
            return "SIG_REGULATE_GENERIC"

    elif source == "SMPDB" and relation == "ANNOTATED_IN_PATHWAY":
        if head.startswith("HMDB"):
            return "METABOLITE_IN_PATHWAY"
        if head and head[0] in {"P", "Q", "O", "A", "B", "H"}:
            return "PROTEIN_IN_PATHWAY"

    elif source == "STITCH":
        action = str(attributes.get("action", "association")).lower().strip()
        evidence = str(attributes.get("evidences", "")).lower()
        score = as_float(attributes.get("score"))
        if "activation" in action:
            base = "DRUG_ACTIVATE"
        elif "inhibition" in action:
            base = "DRUG_INHIBIT"
        elif "binding" in action:
            base = "DRUG_BIND"
        elif "expression" in action:
            base = "DRUG_REGULATE_EXPRESSION"
        elif "catalysis" in action or "reaction" in action:
            base = "DRUG_METABOL_SUBSTRATE"
        else:
            base = "DRUG_CHEMICAL_ASSOC"
        return base + ("_GOLD" if score >= 0.7 and "experimental" in evidence else "_SILVER" if score >= 0.4 else "_BRONZE")

    elif source == "STRING":
        evidence = str(attributes.get("evidence", "")).lower()
        action = str(attributes.get("action", "")).lower()
        if relation == "ACTS_ON":
            if "activation" in action:
                return "STRING_STIMULATE"
            if "inhibition" in action:
                return "STRING_INHIBIT"
            if "ptmod" in action:
                return "STRING_MODIFIES"
            return "STRING_REACTION"
        if "experimental" in evidence:
            base = "STRING_PHYSICAL"
        elif "databases" in evidence:
            base = "STRING_CURATED"
        elif any(term in evidence for term in ("neighborhood", "fusion", "co-ocurrence")):
            base = "STRING_GENOMIC"
        elif "co-expression" in evidence:
            base = "STRING_COEXP"
        else:
            base = "STRING_TEXTMINING"
        score = as_float(attributes.get("score"))
        return base + ("_GOLD" if score >= 0.7 else "_SILVER" if score >= 0.4 else "_BRONZE")

    elif source == "TISSUES":
        score = as_float(attributes.get("score"))
        if score >= 4.0:
            base = "TISSUE_ENRICHED"
        elif score >= 2.5:
            base = "TISSUE_DETECTED_STRONG"
        elif score >= 1.0:
            base = "TISSUE_DETECTED_WEAK"
        else:
            base = "TISSUE_TRACE_LEVEL"
        return base + ("_CONFIRMED" if score >= 4.5 else "_PROBABLE")

    elif source == "UniProt":
        evidence_type = str(attributes.get("evidence_type", "IEA"))
        interaction = str(attributes.get("interaction_type", "")).lower()
        if relation == "ASSOCIATED_WITH":
            if evidence_type in {"EXP", "IDA", "IMP", "IGI", "TAS", "curated"}:
                return "FUNC_VALIDATED_EXP"
            if evidence_type in {"IEA", "ND"}:
                return "FUNC_PREDICTED_AUTO"
            return "FUNC_INFERRED_HOMOLOGY"
        if relation == "CURATED_INTERACTS_WITH":
            if "phosphorylation" in interaction:
                return "SIGNAL_MODIFICATION_PHOSPHO"
            if "cleavage" in interaction:
                return "SIGNAL_PROTEOLYSIS"
            if "direct interaction" in interaction:
                return "PHYSICAL_DIRECT_BIND"
            return "PHYSICAL_ASSOCIATION_GENERIC"
        if relation == "HAS_STRUCTURE":
            return "STRUCT_3D_PDB"
        if relation == "HAS_SEQUENCE":
            return "STRUCT_1D_AA"

    elif source == "bhf-ucl":
        interaction = str(attributes.get("interaction_type", ""))
        value = interaction.lower()
        if any(term in value for term in ("phosphorylation reaction", "ubiquitination reaction", "protein cleavage", "disulfide bond")):
            return "ACTS_BIOCHEMICALLY_WITH"
        if "direct interaction" in value:
            return "INTERACTS_DIRECTLY_WITH"
        if "physical association" in value:
            return "ASSOCIATES_PHYSICALLY_WITH"
        if "proximity" in value or "association" in value:
            return "FUNCTIONALLY_ASSOCIATED_WITH"
        if value.strip() == "colocalization":
            return "COLOCALIZES_WITH"

    elif source == "matrixdb" and relation == "CURATED_INTERACTS_WITH":
        return interaction_relation("ECM", attributes.get("interaction_type"), attributes.get("score"))

    return relation


def group_inputs(input_dir: Path) -> tuple[dict[str, list[Path]], list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    unsupported: list[Path] = []
    for path in sorted(input_dir.glob("*.txt")):
        if not path.is_file():
            continue
        source = canonical_source(path.stem)
        if source is None:
            unsupported.append(path)
        else:
            grouped[source].append(path)
    return grouped, unsupported


def process_directory(input_dir: Path, output_dir: Path) -> None:
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped, unsupported = group_inputs(input_dir)
    for path in unsupported:
        print(f"Skipping unsupported CKG source without a benchmark preprocessing rule: {path.name}")

    for source, input_paths in sorted(grouped.items()):
        output_path = output_dir / f"{source}.txt"
        written = 0
        invalid = 0
        with output_path.open("w", encoding="utf-8", newline="") as output_stream:
            writer = csv.writer(output_stream, delimiter="\t")
            for input_path in input_paths:
                with input_path.open("r", encoding="utf-8", newline="") as input_stream:
                    for row in csv.reader(input_stream, delimiter="\t"):
                        if len(row) < 4 or (row[0] == "start_node" and row[1] == "rel_type"):
                            continue
                        head, relation, tail = (value.strip() for value in row[:3])
                        if not head or not relation or not tail:
                            invalid += 1
                            continue
                        try:
                            attributes = json.loads(row[3])
                        except json.JSONDecodeError:
                            invalid += 1
                            continue
                        if not isinstance(attributes, dict):
                            invalid += 1
                            continue
                        writer.writerow(
                            [head, refine_relation(source, head, relation, attributes), tail]
                        )
                        written += 1
        print(
            f"{source}: wrote {written:,} triples from {len(input_paths)} file(s); "
            f"skipped {invalid:,} invalid rows"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path, help="four-column CKG export directory")
    parser.add_argument("output_dir", type=Path, help="three-column output directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_directory(args.input_dir, args.output_dir)
