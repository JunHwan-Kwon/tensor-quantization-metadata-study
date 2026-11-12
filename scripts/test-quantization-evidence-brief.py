#!/usr/bin/env python3
"""Reject evidence drift and malformed layout inputs; no model downloads."""
import copy
import importlib.util
import io
import hashlib
import json
import struct
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.replace('-', '_'), ROOT / 'scripts' / filename)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


brief = module('build-quantization-evidence-brief.py')
layout_module = module('measure-gguf-layout.py')


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


class EvidenceDriftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.left = load(brief.GGUF + '/results/ibm-q4km-tensor-inventory.json')
        cls.right = load(brief.GGUF + '/results/bartowski-q4km-tensor-inventory.json')
        cls.comparison = load(brief.GGUF + '/results/tensor-assignment-comparison.json')
        cls.layout = load(brief.GGUF + '/results/file-layout.json')

    def test_original_inputs(self):
        self.assertEqual(len(brief.compare_pair(self.left, self.right, self.comparison)), 100)
        result = brief.validate_layout(self.layout, [self.left, self.right], ROOT)
        self.assertEqual((result['commonEqualCount'], len(result['rightOnlyKeys'])), (35, 10))

    def test_payload_formula_drift(self):
        changed = copy.deepcopy(self.left)
        changed['tensors'][0]['byteLength'] += 1
        with self.assertRaisesRegex(ValueError, 'Encoding/payload'):
            brief.validate_inventory(changed)

    def test_duplicate_name(self):
        changed = copy.deepcopy(self.left)
        changed['tensors'][1]['name'] = changed['tensors'][0]['name']
        with self.assertRaisesRegex(ValueError, 'Duplicate tensor'):
            brief.validate_inventory(changed)

    def test_missing_index(self):
        changed = copy.deepcopy(self.left)
        changed['tensors'][1]['index'] = changed['tensors'][0]['index']
        with self.assertRaisesRegex(ValueError, 'index coverage'):
            brief.validate_inventory(changed)

    def test_wrong_shape(self):
        changed = copy.deepcopy(self.left)
        changed['tensors'][0]['shape'][0] += 1
        with self.assertRaisesRegex(ValueError, 'Shape/element'):
            brief.validate_inventory(changed)

    def test_stale_changed_tensor_list(self):
        changed = copy.deepcopy(self.comparison)
        changed['changedTensors'].pop()
        with self.assertRaisesRegex(ValueError, 'Changed tensor rows'):
            brief.compare_pair(self.left, self.right, changed)

    def test_pair_from_another_file(self):
        changed = copy.deepcopy(self.comparison)
        changed['subjects'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Pair source binding'):
            brief.compare_pair(self.left, self.right, changed)

    def test_layout_from_another_file(self):
        changed = copy.deepcopy(self.layout)
        changed['subjects'][0]['artifactSha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Layout artifact binding'):
            brief.validate_layout(changed, [self.left, self.right], ROOT)

    def test_stale_source_digest(self):
        changed = copy.deepcopy(self.layout)
        changed['sources'][0]['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Layout source hash'):
            brief.validate_layout(changed, [self.left, self.right], ROOT)

    def test_metadata_span_gap(self):
        changed = copy.deepcopy(self.layout)
        changed['subjects'][0]['metadata'][0]['entryStart'] += 1
        with self.assertRaisesRegex(ValueError, 'Noncontiguous metadata'):
            brief.validate_layout(changed, [self.left, self.right], ROOT)

    def test_inline_metadata_value_drift(self):
        changed = copy.deepcopy(self.layout)
        row = next(r for r in changed['subjects'][0]['metadata'] if 'value' in r)
        row['value'] = 'unexpected'
        with self.assertRaisesRegex(ValueError, 'inline value digest'):
            brief.validate_layout(changed, [self.left, self.right], ROOT)

    def test_bad_region_total(self):
        changed = copy.deepcopy(self.layout)
        changed['subjects'][0]['regions']['alignmentPaddingBytes'] += 1
        with self.assertRaisesRegex(ValueError, 'regions do not sum'):
            brief.validate_layout(changed, [self.left, self.right], ROOT)


class MetadataPresentationTests(unittest.TestCase):
    def test_paths_are_identified_without_republishing_values(self):
        examples = [
            ("quantize.imatrix.file", "relative/matrix.gguf"),
            ("quantize.imatrix.dataset", "calibration.txt"),
            ("example.path", "/producer/work/matrix.gguf"),
            ("example.path", "C:\\build\\matrix.gguf"),
            ("example.path", "file:///producer/matrix.gguf"),
            ("example.paths", ["public-label", "/producer/work/matrix.gguf"]),
        ]
        for key, value in examples:
            with self.subTest(key=key, value=value):
                row = layout_module.metadata_identity(key, ["STRING"], 24, 100, value)
                self.assertNotIn("value", row)
                self.assertEqual(row["valueOmission"], "producer_file_path" if key.startswith("quantize.imatrix.") else "path_like_value")
                encoded = layout_module.canonical(value)
                self.assertEqual(row["canonicalValueSha256"], hashlib.sha256(encoded).hexdigest())
                self.assertEqual(row["canonicalValueBytes"], len(encoded))
                self.assertEqual((row["entryStart"], row["entryBytes"]), (24, 100))

    def test_public_identifiers_and_numeric_values_are_preserved(self):
        for value in ("Granite 4.2 3b", "https://huggingface.co/ibm-granite", 573, [1, 2]):
            with self.subTest(value=value):
                row = layout_module.metadata_identity("example", ["STRING"], 24, 100, value)
                self.assertEqual(row["value"], value)
                self.assertNotIn("valueOmission", row)

    def test_large_values_have_an_explicit_omission_reason(self):
        row = layout_module.metadata_identity("tokenizer.example", ["ARRAY", "STRING"], 24, 400, ["/", "x" * 300])
        self.assertNotIn("value", row)
        self.assertEqual(row["valueOmission"], "encoded_value_exceeds_256_bytes")

    def test_published_layout_has_no_inline_producer_paths(self):
        layout = load(brief.GGUF + '/results/file-layout.json')
        for subject in layout["subjects"]:
            for row in subject["metadata"]:
                with self.subTest(key=row["key"]):
                    if "value" in row:
                        projected = layout_module.metadata_identity(
                            row["key"], row["types"], row["entryStart"], row["entryBytes"], row["value"])
                        self.assertIn("value", projected)
                    else:
                        self.assertIn(row.get("valueOmission"), (
                            "producer_file_path", "path_like_value", "encoded_value_exceeds_256_bytes"))


class CursorTests(unittest.TestCase):
    def test_truncated_integer(self):
        with self.assertRaisesRegex(ValueError, 'boundary'):
            layout_module.Cursor(io.BytesIO(b'123'), 3).number('Q')

    def test_out_of_bounds_string(self):
        data = struct.pack('<Q', 200) + b'x'
        with self.assertRaisesRegex(ValueError, 'boundary'):
            layout_module.Cursor(io.BytesIO(data), len(data)).string()

    def test_unknown_metadata_type(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported GGUF'):
            layout_module.Cursor(io.BytesIO(b''), 0).value(99)

    def test_impossible_array_count(self):
        data = struct.pack('<IQ', 8, 2**63)
        with self.assertRaisesRegex(ValueError, 'Array count'):
            layout_module.Cursor(io.BytesIO(data), len(data)).value(9)


if __name__ == '__main__':
    unittest.main()
