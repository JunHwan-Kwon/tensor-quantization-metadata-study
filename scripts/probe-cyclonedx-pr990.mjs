import { createHash } from "node:crypto";
import { createRequire } from "node:module";
import { gunzipSync } from "node:zlib";
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import Ajv2020 from "ajv/dist/2020.js";

const require = createRequire(import.meta.url);
const draft7MetaSchema = require("ajv/dist/refs/json-schema-draft-07.json");

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const PROBE_ROOT = join(
  ROOT,
  "conformance",
  "cyclonedx-2.0",
  "quantization-ownership-probes",
);
const MANIFEST_PATH = join(PROBE_ROOT, "manifest.json");
const RESULT_PATH = join(ROOT, "data", "cyclonedx-pr990-validation-result.json");
const PREFIX = "cdx:ai-ml:model:parameter:quantization:";
const ROOT_SCHEMA_ID = "https://cyclonedx.org/schema/2.0/cyclonedx-2.0.schema.json";

function fail(message) {
  throw new Error(message);
}

function sha256(bytes) {
  return createHash("sha256").update(bytes).digest("hex");
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function stableStringify(value) {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map(stableStringify).join(",")}]`;
  }
  return `{${Object.keys(value)
    .sort()
    .map((key) => `${JSON.stringify(key)}:${stableStringify(value[key])}`)
    .join(",")}}`;
}

function loadSchemaSet(manifest) {
  const archivePath = join(ROOT, manifest.schema_source.archive);
  const compressed = readFileSync(archivePath);
  if (sha256(compressed) !== manifest.schema_source.archive_sha256) {
    fail("Pinned CycloneDX schema archive SHA-256 mismatch");
  }
  const archive = JSON.parse(gunzipSync(compressed).toString("utf8"));
  if (archive.schema !== "tensor_quantization_metadata_study.pinned_json_schema_set.v1") {
    fail("Unexpected pinned schema-set contract");
  }
  if (archive.repository !== `https://github.com/${manifest.schema_source.repository}`) {
    fail("Pinned schema repository mismatch");
  }
  if (archive.commit !== manifest.schema_source.commit) {
    fail("Pinned schema commit mismatch");
  }
  if (archive.files.length !== manifest.schema_source.source_file_count) {
    fail("Pinned schema file count mismatch");
  }

  const files = archive.files.map((entry) => {
    const bytes = Buffer.from(entry.content_base64, "base64");
    if (sha256(bytes) !== entry.sha256) {
      fail(`Pinned schema member SHA-256 mismatch: ${entry.path}`);
    }
    return { ...entry, schema: JSON.parse(bytes.toString("utf8")) };
  });
  return { archive, files };
}

function buildValidator(files, manifest) {
  const ajv = new Ajv2020({
    allErrors: true,
    strict: false,
    validateFormats: false,
  });
  ajv.addMetaSchema(draft7MetaSchema);
  const excluded = new Set(manifest.schema_source.validation_excluded_paths);
  const validationFiles = files.filter((entry) => !excluded.has(entry.path));
  if (validationFiles.length !== manifest.schema_source.validation_file_count) {
    fail("Pinned modular validation file count mismatch");
  }
  for (const entry of validationFiles) {
    if (entry.path.includes("bundled")) {
      fail(`Bundled schema entered modular validation graph: ${entry.path}`);
    }
    if (!entry.schema.$id) {
      fail(`Pinned modular schema has no $id: ${entry.path}`);
    }
    ajv.addSchema(entry.schema);
    if (entry.path === "schema/spdx.schema.json") {
      ajv.addSchema({
        ...entry.schema,
        $id: "https://cyclonedx.org/schema/2.0/spdx.schema.json",
      });
    }
    if (entry.path === "schema/cryptography-defs.schema.json") {
      ajv.addSchema({
        ...entry.schema,
        $id: "https://cyclonedx.org/schema/2.0/cryptography-defs.schema.json",
      });
    }
  }
  const validate = ajv.getSchema(ROOT_SCHEMA_ID);
  if (!validate) {
    fail("Pinned modular CycloneDX root schema was not registered");
  }
  return { validate, validationFiles };
}

function verifyTaxonomySource(manifest) {
  const taxonomyPath = join(ROOT, manifest.taxonomy_source.vendored_path);
  const bytes = readFileSync(taxonomyPath);
  if (sha256(bytes) !== manifest.taxonomy_source.sha256) {
    fail("Pinned property-taxonomy SHA-256 mismatch");
  }
  const text = bytes.toString("utf8");
  const requiredFragments = [
    "This property MUST be present whenever any other `quantization` sub-property is used.",
    "For per-tensor quantization, value is a string containing a single decimal number",
    "Required when `quantization:granularity` is `per-axis`",
    "`_undefined:<NAME>` | `<NAME>` placeholder, used to identify a quantization scheme not yet listed",
  ];
  for (const fragment of requiredFragments) {
    if (!text.includes(fragment)) {
      fail(`Pinned taxonomy no longer contains required text: ${fragment}`);
    }
  }
  return bytes;
}

function parameterFrom(document) {
  return document.components?.[0]?.modelProperties?.inputs?.[0] ?? null;
}

function taxonomyProperties(parameter) {
  const rows = parameter?.properties ?? [];
  const selected = rows.filter((row) => row.name?.startsWith(PREFIX));
  const values = new Map();
  for (const row of selected) {
    const name = row.name.slice(PREFIX.length);
    if (values.has(name)) {
      return { status: "invalid_duplicate_property", values };
    }
    values.set(name, row.value);
  }
  return { status: null, values };
}

function parseDecimalNumber(value) {
  if (
    typeof value !== "string"
    || !/^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/.test(value)
  ) {
    return null;
  }
  return value;
}

function parseInteger(value) {
  if (typeof value !== "string" || !/^-?(0|[1-9][0-9]*)$/.test(value)) {
    return null;
  }
  try {
    return BigInt(value);
  } catch {
    return null;
  }
}

function parseNumericArray(value, integersOnly) {
  if (typeof value !== "string") {
    return null;
  }
  const trimmed = value.trim();
  if (!trimmed.startsWith("[") || !trimmed.endsWith("]")) {
    return null;
  }
  const body = trimmed.slice(1, -1).trim();
  if (!body) {
    return null;
  }
  const values = body.split(",").map((item) => item.trim());
  const pattern = integersOnly
    ? /^-?(?:0|[1-9][0-9]*)$/
    : /^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?$/;
  if (!values.every((item) => pattern.test(item))) {
    return null;
  }
  return values;
}

function taxonomyStatus(document) {
  const parameter = parameterFrom(document);
  if (!parameter) {
    return "legacy_aggregate_placement";
  }
  const parsed = taxonomyProperties(parameter);
  if (parsed.status) {
    return parsed.status;
  }
  const properties = parsed.values;
  if (properties.size === 0) {
    return "not_used";
  }
  if (!properties.has("scheme")) {
    return "invalid_missing_scheme";
  }
  const scheme = properties.get("scheme");
  const customScheme = typeof scheme === "string"
    && scheme.startsWith("_undefined:")
    && scheme.length > "_undefined:".length;
  if (
    scheme !== "affine_asymmetric"
    && scheme !== "affine_symmetric"
    && !customScheme
  ) {
    return "invalid_unknown_scheme";
  }
  if (
    (scheme === "affine_asymmetric" || scheme === "affine_symmetric")
    && !properties.has("scale")
  ) {
    return "invalid_missing_scale";
  }
  if (scheme === "affine_asymmetric" && !properties.has("zeroPoint")) {
    return "invalid_missing_zero_point";
  }

  const granularity = properties.get("granularity") ?? "per-tensor";
  if (granularity !== "per-tensor" && granularity !== "per-axis") {
    return "invalid_granularity";
  }
  if (granularity === "per-tensor") {
    const scale = properties.has("scale")
      ? parseDecimalNumber(properties.get("scale"))
      : null;
    if (properties.has("scale") && scale === null) {
      return "invalid_per_tensor_vector";
    }
    if (properties.has("zeroPoint") && parseInteger(properties.get("zeroPoint")) === null) {
      return "invalid_per_tensor_vector";
    }
    return "valid";
  }

  const axis = parseInteger(properties.get("axis"));
  if (axis === null || axis < 0n) {
    return "invalid_per_axis_axis";
  }
  const scales = properties.has("scale")
    ? parseNumericArray(properties.get("scale"), false)
    : null;
  if (properties.has("scale") && !scales) {
    return "invalid_per_axis_scale";
  }
  if (properties.has("zeroPoint")) {
    const zeroPoints = parseNumericArray(properties.get("zeroPoint"), true);
    if (!zeroPoints || (scales && zeroPoints.length !== scales.length)) {
      return "invalid_per_axis_zero_point";
    }
  }
  return "valid";
}

function verifyTaxonomyCheckerBoundary() {
  const document = (scheme, properties = []) => ({
    components: [{
      modelProperties: {
        inputs: [{
          properties: [
            { name: `${PREFIX}scheme`, value: scheme },
            ...properties,
          ],
        }],
      },
    }],
  });
  const cases = [
    {
      id: "named-custom-scheme",
      document: document("_undefined:vendor_scheme"),
      expected: "valid",
    },
    {
      id: "empty-custom-scheme-name",
      document: document("_undefined:"),
      expected: "invalid_unknown_scheme",
    },
    {
      id: "ignored-per-tensor-axis",
      document: document("affine_asymmetric", [
        { name: `${PREFIX}granularity`, value: "per-tensor" },
        { name: `${PREFIX}axis`, value: "0" },
        { name: `${PREFIX}scale`, value: "0.5" },
        { name: `${PREFIX}zeroPoint`, value: "0" },
      ]),
      expected: "valid",
    },
  ];
  for (const row of cases) {
    const observed = taxonomyStatus(row.document);
    if (observed !== row.expected) {
      fail(`Taxonomy checker boundary mismatch: ${row.id}`);
    }
  }
  return cases.length;
}

function ownershipStatus(document) {
  const parameter = parameterFrom(document);
  if (!parameter) {
    return "legacy_aggregate";
  }
  const typed = parameter.quantization ?? null;
  const properties = taxonomyProperties(parameter).values;
  if (!typed && properties.size === 0) {
    return "unrepresented";
  }
  if (typed && properties.size === 0) {
    return "typed_core_only";
  }
  const duplicatedNames = ["scheme", "granularity", "axis"]
    .filter((name) => properties.has(name));
  if (typed && duplicatedNames.length === 0) {
    return "typed_core_with_numeric_extension";
  }

  const schemeMap = new Map([
    ["affine", "affine_asymmetric"],
    ["symmetric", "affine_symmetric"],
  ]);
  const granularityMap = new Map([
    ["per-tensor", "per-tensor"],
    ["per-channel", "per-axis"],
  ]);
  const equivalent = duplicatedNames.every((name) => {
    if (name === "scheme") {
      return schemeMap.get(typed?.scheme) === properties.get(name);
    }
    if (name === "granularity") {
      return granularityMap.get(typed?.granularity) === properties.get(name);
    }
    return String(typed?.axis) === properties.get(name);
  });
  return equivalent ? "duplicated_equivalent" : "contradictory_duplicate";
}

function schemaFailureSummary(errors) {
  return [...(errors ?? [])]
    .map((error) => ({
      instance_pointer: error.instancePath || "",
      keyword: error.keyword,
    }))
    .sort((left, right) => stableStringify(left).localeCompare(stableStringify(right)));
}

function buildResult(
  manifest,
  files,
  validationFiles,
  taxonomyBytes,
  validate,
  taxonomyCheckerUnitCaseCount,
) {
  const probes = manifest.probes.map((probe) => {
    const path = join(PROBE_ROOT, probe.path);
    const bytes = readFileSync(path);
    const document = JSON.parse(bytes.toString("utf8"));
    const schemaValid = Boolean(validate(document));
    const schemaFailures = schemaFailureSummary(validate.errors);
    const observedTaxonomyStatus = taxonomyStatus(document);
    const observedOwnershipStatus = ownershipStatus(document);

    if (schemaValid !== probe.expected_schema_valid) {
      fail(`Schema expectation mismatch: ${probe.id}`);
    }
    if (observedTaxonomyStatus !== probe.expected_taxonomy_status) {
      fail(`Taxonomy expectation mismatch: ${probe.id}`);
    }
    if (observedOwnershipStatus !== probe.expected_ownership_status) {
      fail(`Ownership expectation mismatch: ${probe.id}`);
    }
    return {
      id: probe.id,
      path: relative(ROOT, path).replaceAll("\\", "/"),
      sha256: sha256(bytes),
      schema_valid: schemaValid,
      schema_failures: schemaFailures,
      taxonomy_status: observedTaxonomyStatus,
      ownership_status: observedOwnershipStatus,
    };
  });

  const result = {
    schema: "tensor_quantization_metadata_study.cyclonedx_pr990_validation.v1.1",
    claim_boundary: [
      "Schema validity is measured against a 26-file modular graph drawn from the 28-file pinned PR #990 source set; the two bundled schemas are retained only for provenance.",
      "Taxonomy status is a deterministic check of the pinned PR #175 text; it is not JSON Schema validation.",
      "Taxonomy status does not add scale-positivity or integer-storage-range requirements that are absent from the pinned text.",
      "The probes assess placement and ownership semantics, not runtime behavior or model quality.",
    ],
    validators: {
      primary: {
        implementation: "Ajv",
        version: "8.20.0",
        dialect: "https://json-schema.org/draft/2020-12/schema",
      },
      independent: {
        implementation: "python-jsonschema",
        required_version: "4.26.0",
        script: "scripts/verify-cyclonedx-pr990-files.py",
      },
    },
    sources: {
      schema: {
        repository: manifest.schema_source.repository,
        pull_request: manifest.schema_source.pull_request,
        commit: manifest.schema_source.commit,
        archive_sha256: manifest.schema_source.archive_sha256,
        source_member_count: files.length,
        validation_member_count: validationFiles.length,
        validation_excluded_paths: manifest.schema_source.validation_excluded_paths,
      },
      taxonomy: {
        repository: manifest.taxonomy_source.repository,
        pull_request: manifest.taxonomy_source.pull_request,
        commit: manifest.taxonomy_source.commit,
        sha256: sha256(taxonomyBytes),
      },
    },
    observations: {
      typed_only_schema_valid: probes.find((probe) => probe.id === "typed-only").schema_valid,
      typed_numeric_extension_schema_valid: probes.find((probe) => probe.id === "taxonomy-extension").schema_valid,
      contradictory_duplicate_schema_valid: probes.find((probe) => probe.id === "contradiction").schema_valid,
      semantic_equivalence_enforced_by_schema: false,
      taxonomy_checker_unit_case_count: taxonomyCheckerUnitCaseCount,
    },
    probes,
    hash_contract: {
      algorithm: "SHA-256",
      canonicalization: "UTF-8 JSON with recursively lexicographic object keys and compact separators",
      excluded_pointer: "/ledger_sha256",
    },
  };
  result.ledger_sha256 = sha256(Buffer.from(stableStringify(result), "utf8"));
  return result;
}

function main() {
  const mode = process.argv[2];
  if (mode !== "--write" && mode !== "--check") {
    fail("Usage: node scripts/probe-cyclonedx-pr990.mjs --write|--check");
  }
  const manifest = readJson(MANIFEST_PATH);
  const { files } = loadSchemaSet(manifest);
  const taxonomyBytes = verifyTaxonomySource(manifest);
  const taxonomyCheckerUnitCaseCount = verifyTaxonomyCheckerBoundary();
  const { validate, validationFiles } = buildValidator(files, manifest);
  const result = buildResult(
    manifest,
    files,
    validationFiles,
    taxonomyBytes,
    validate,
    taxonomyCheckerUnitCaseCount,
  );
  const serialized = `${JSON.stringify(result, null, 2)}\n`;

  if (mode === "--write") {
    writeFileSync(RESULT_PATH, serialized, "utf8");
  } else {
    const existing = readFileSync(RESULT_PATH, "utf8");
    if (existing !== serialized) {
      fail("CycloneDX PR #990 validation result is stale; run npm run generate:cyclonedx-pr990");
    }
  }
  process.stdout.write(`${JSON.stringify({ status: "pass", probe_count: result.probes.length }, null, 2)}\n`);
}

main();
