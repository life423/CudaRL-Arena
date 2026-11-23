# Research evidence publication

The original local diagnostic archives and their recorded hashes remain
unchanged. Publication does not rerun a study, change its outcome or rewrite
the original verification bindings.

The v1 evidence archive contains the original160members and is published with
its original manifest. The v2 archive originally contained234members, including
a copied installed CUDA compiler executable. That compiler is not needed in
the repository. The exact original v2 archive/manifest stay local and ignored.

The public v2 copy is explicitly a233-member subset:

- [evidence-public.zip](../assets/research-evidence/frozen-interface-comparison-2026-11-23-v2/evidence-public.zip)
- [manifest-public.json](../assets/research-evidence/frozen-interface-comparison-2026-11-23-v2/manifest-public.json)

Public archive SHA256:
7c587b8dc38b80b626f1f6989640834f3b6bb7c2211a67b319a66bca4dbceb15.
Size:21499721bytes. Every included member was read back and checked against
its original SHA256. The manifest identifies the original archive hash and
lists every included and excluded member with exact byte size and checksum.

The sole excluded member is the19793008-byte nvcc.exe snapshot, SHA256
537980c3892ecddbb278de6497f849cd896f9e359a5bf0001167d82f3a3cc2e3.
Its original location and hash remain recorded. Install the matching CUDA
toolchain separately to satisfy that original external-tool binding. The
public subset is not described as the complete original archive.

Original v2 local archive SHA256:
e6af9d8ea725a7b8681183e61bdb6b08f602f2f73e11e7baff05cff3a2e18dc7.
Original v1 archive SHA256:
2af9f92a0b9597e2e2c006499fa428d3eb10f70d5d7f09e5c7dd5650e7f08dc0.

Failed runs, zero-time arithmetic traces, checkpoint/force/source bindings and
closed-budget outcomes remain separate from the untimed Godot product change.
Hosted CPU CI does not claim execution of the GPU research studies. It also
builds/runs the new untimed bookkeeping checker:333assertions, zero physical
integration or optimizer work. Local Godot/module and bounded product evidence
remain documented in the corresponding verification reports.
