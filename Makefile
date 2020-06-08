.PHONY: help
help: Makefile
	@ sed -n 's/^##//p' $<

## clean_chkpts: Delete all trained chkpts
clean_chkpts:
	rm ./checkpoints/*.pth

## clean_mlruns: Delete all ml experiments
clean_mlruns:
	rm -rf ./mlruns/*