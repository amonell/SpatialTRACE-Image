"""Lightweight layout regression tests; no scientific data or fitting required."""
import json
from paper_paths import Path, lock_record, rendering_script

import unittest

REPO = Path(__file__).resolve().parents[2]
PACKAGES = ["Figure_2", "Figure_2_Extended", "Figure_2_Extended_2", "Figure_3_Extended"]


def check_original_geometry(package):
    layout = json.loads((REPO / package / "assembly/layout.json").read_text())
    source = json.loads((REPO / package / "assembly/source_registration.json").read_text())
    assert source["reference_sha256"] == layout["reference_sha256"]
    images = {item["id"]:item["rect_pt"] for item in source["images"]}
    paths = {item["index"]:item["rect_pt"] for item in source["paths"]}
    assert len(layout["source_anchors"]) >= 12
    for anchor in layout["source_anchors"]:
        if anchor["source"].startswith("page/"):
            assert max(abs(a-b) for a,b in zip(anchor["rect_pt"],images[anchor["source"]],strict=True)) < 1e-6
        else:
            index = int(anchor["source"].split(":")[1])
            expected = paths[index]
            # The stored vertical spine establishes x, top and plot height;
            # its paired horizontal spine establishes width in the manifest.
            assert max(abs(anchor["rect_pt"][i]-expected[i]) for i in (0,1,3)) < 1e-6
            x,y,width,height = anchor["rect_pt"]
            assert any(abs(box[0]-x)<1e-6 and abs(box[1]-(y+height))<1e-3
                       and abs(box[2]-width)<1e-6 and box[3]<.01
                       for box in paths.values())


def check_fresh_placement(package):
    layout = json.loads((REPO / package / "assembly/layout.json").read_text())
    assert layout["page_size_pt"] == [595.276, 841.89]
    for element in layout["elements"]:
        assert element["panel"] in layout["panels"]
        if element["kind"] != "pdf":
            assert element["font"].startswith("Arial")
            continue
        assert not Path(element["path"]).is_absolute()
        stem = Path(element["path"]).stem
        spec = layout["render_specs"][stem]
        assert element["rect_pt"][2:] == spec["size_pt"]
        assert element["fit"] == "exact"


def check_fixed_export():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from Figure_2.assembly.fixed_render import configure
    figure = Figure(figsize=(4,3))
    axis = figure.add_axes([.2,.2,.6,.6], label="<colorbar>")
    axis.set_title("A fixed title")
    axis.set_box_aspect(20)
    spec = {"size_pt":[100,100], "axes":[{"rect_pt":[20,15,10,70]}], "title_size_pt":6.84}
    configure(figure, spec)
    assert abs(axis.title.get_fontsize()-6.84) < 1e-9
    assert max(abs(a-b) for a,b in zip(axis.get_position().bounds,(.2,.15,.1,.7),strict=True)) < 1e-9


class FixedLayoutTests(unittest.TestCase):
    def test_original_geometry(self):
        for package in PACKAGES:
            with self.subTest(package=package):
                check_original_geometry(package)

    def test_fresh_placement(self):
        for package in PACKAGES:
            with self.subTest(package=package):
                check_fresh_placement(package)

    def test_font_and_colorbar_regression(self):
        check_fixed_export()

    def test_manual_art_partition(self):
        layout = json.loads((REPO / "Figure_2/assembly/layout.json").read_text())
        groups = layout["static_groups"]
        partition = [index for values in groups["by_panel"].values() for index in values]
        self.assertEqual(len(partition), len(set(partition)))
        self.assertEqual(set(partition), set(groups["keep_paint_indices"]))

    def test_schematic_components_have_real_generators(self):
        panel_map = json.loads((REPO / "Figure_2/panel_order.json").read_text())
        panel = panel_map["panels"][0]
        self.assertEqual(len(panel["components"]), 6)
        for script in panel["render_scripts"]:
            self.assertTrue((REPO / "Figure_2" / script).is_file())
        layout = json.loads((REPO / "Figure_2/assembly/layout.json").read_text())
        self.assertTrue(set(panel["components"]).issubset(layout["render_specs"]))

    def test_prepared_roots_are_independent(self):
        import subprocess
        import sys
        import shlex
        from Figure_3_Extended.run_all_panels import command_plan
        output = Path("/tmp/unused-panel-output-for-dry-run")
        prepared = Path("/tmp/unused-prepared-inputs-for-dry-run")
        plan = command_plan(output, python=sys.executable, overwrite=False, prepared_root=prepared)
        self.assertIn(str(prepared / "paired_views.npz"), plan["A"])
        self.assertNotIn(str(output / "stored_outputs/paired_view_repair_v1/paired_views.npz"), plan["A"])
        result = subprocess.run([sys.executable, str(REPO / "Figure_2/run_all_panels.py"),
            "--output-root", str(output), "--prepared-root", str(prepared), "--panels", "A", "--dry-run"],
            check=True, text=True, capture_output=True)
        commands = [shlex.split(line) for line in result.stdout.splitlines() if "figure_script/" in line]
        self.assertEqual(len(commands), 2)
        self.assertIn(str(prepared / "section_cells.parquet"), commands[0])
        self.assertIn(str(prepared / "day90_zoom_cells.parquet"), commands[1])
        self.assertFalse(any("stored_outputs" in value and str(output) in value for cmd in commands for value in cmd))

    def test_modified_graph_input_is_rejected(self):
        import tempfile
        from Figure_2.common import require_graph_inputs
        manifest = Path("/mnt/sata1/Analysis_Alex/Desktop_2/spatial_axes_manuscript/Figure_2/stored_outputs/shared_graph_components/graph_component_inputs_manifest.json")
        if not manifest.exists():
            self.skipTest("Mounted compact input manifest is unavailable")
        with tempfile.TemporaryDirectory(prefix="figure2-hash-rejection-") as folder:
            altered = Path(folder) / "section_cells.parquet"
            altered.write_bytes(b"deliberately invalid test input")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                require_graph_inputs(manifest, [altered])


if __name__ == "__main__":
    unittest.main()
