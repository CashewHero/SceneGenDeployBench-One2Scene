# Third-party notices

The One2Scene source tree does not include a repository-level license file at the reviewed upstream commit. Individual files under `src/model/encoder/heads/` and `src/model/encoder/backbone/croco/` state that they are licensed under CC BY-NC-SA 4.0 for non-commercial use. Other embedded components carry their own file-level notices.

The One2Scene model checkpoints are distributed separately through `mutou0308/One2Scene` on Hugging Face. The runner downloads the scaffold checkpoint at runtime and does not include it in the container image.

Review the upstream source and checkpoint terms before redistributing an image or using it outside research and other permitted non-commercial contexts.
