# V3.2 CompShare capacity receipt (2026-09-05)

The retained CompShare instance `uhost-1utsjo3ep1jz` was checked before any
GPU start.  Its original stopped profile (2 CPU, 4 GiB, 160 GiB) was not
usable for the corrected prepared folds: each `.pt` is approximately 10--13
GiB and the complete G0/G1/G2 input tree is approximately 170 GiB.

The instance was resized while still stopped to the provider-supported 4090
profile:

```text
CPU: 16
RAM: 64 GiB
GPU: 1 x RTX 4090
boot disk: 250 GiB CLOUD_SSD
state after resize: Stopped
```

The provider query reports approximately 2.05 CNY/hour for compute and 0.07
CNY/hour for the stopped 250-GiB boot disk.  No GPU process, optimizer, or
training job was started by this resize.  The resize is capacity preparation
only; the launch gate still requires the corrected 15-fold rebind receipt,
code/input hashes, and a valid training preflight.

