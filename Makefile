PYTHON ?= python3
MPLBACKEND ?= Agg
MPLCONFIGDIR ?= .mplconfig

.PHONY: canonical_release
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
