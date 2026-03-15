PYTHON ?= python3
MPLBACKEND ?= Agg
MPLCONFIGDIR ?= .mplconfig

.PHONY: canonical_release rich_release parity_map
canonical_release:
	git clean -xfd
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) \
	$(PYTHON) -m scenarios.runner \
		--basis circular \
		--xpd-matrix-source J \
		--physics-validation-mode \
		--release-mode \
		--model-compare \
		--output outputs/rt_dataset_canonical_release.h5 \
		--plots-dir outputs/plots_canonical_release \
		--report outputs/report_canonical_release.md \
		--nf 1024

rich_release:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) \
	$(PYTHON) -m scenarios.runner \
		--rich-mode \
		--scenario-ids A5 \
		--basis circular \
		--xpd-matrix-source J \
		--physics-validation-mode \
		--model-compare \
		--materials-db materials/materials_db.json \
		--material-dispersion on \
		--max-bounce-override 4 \
		--diffuse on \
		--diffuse-model directive \
		--diffuse-factor 0.45 \
		--diffuse-lobe-alpha 8.0 \
		--diffuse-rays-per-hit 2 \
		--min-path-power-db -130 \
		--max-paths-per-case 256 \
		--output outputs/rt_dataset_rich_release.h5 \
		--plots-dir outputs/plots_rich_release \
		--report outputs/report_rich_release.md \
		--nf 256

parity_map:
	MPLBACKEND=$(MPLBACKEND) MPLCONFIGDIR=$(MPLCONFIGDIR) \
	$(PYTHON) scripts/sweep_parity_map.py \
		--output-json outputs/parity_collapse_map.json \
		--out-dir outputs/plots_parity_map \
		--tmp-dir outputs/parity_map_tmp
