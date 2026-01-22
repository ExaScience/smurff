.PHONY: all alpine manylinux manylinux-aarch64 musllinux ubuntu clean push push-manylinux push-manylinux-aarch64 push-musllinux

MANYLINUX_X86_64_IMAGE = vanderaa/manylinux_2_28_x86_64_smurff
MANYLINUX_AARCH64_IMAGE = vanderaa/manylinux_2_28_aarch64_smurff
MUSLLINUX_X86_64_IMAGE = vanderaa/musllinux_1_2_x86_64_smurff

all: alpine manylinux manylinux-aarch64 musllinux ubuntu

push: push-manylinux push-manylinux-aarch64 push-musllinux

alpine:
	docker build -f Dockerfile.alpine -t smurff:alpine .

manylinux:
	docker build -f Dockerfile.manylinux -t $(MANYLINUX_X86_64_IMAGE) .

push-manylinux:
	docker push $(MANYLINUX_X86_64_IMAGE)

manylinux-aarch64:
	docker build -f Dockerfile.manylinux.aarch64 -t $(MANYLINUX_AARCH64_IMAGE) .

push-manylinux-aarch64:
	docker push $(MANYLINUX_AARCH64_IMAGE)

musllinux:
	docker build -f Dockerfile.musllinux -t $(MUSLLINUX_X86_64_IMAGE) .

push-musllinux:
	docker push $(MUSLLINUX_X86_64_IMAGE)

ubuntu:
	docker build -f Dockerfile.ubuntu -t smurff:ubuntu .

clean:
	docker rmi smurff:alpine $(MANYLINUX_X86_64_IMAGE) $(MANYLINUX_AARCH64_IMAGE) $(MUSLLINUX_X86_64_IMAGE) smurff:ubuntu 2>/dev/null || true
