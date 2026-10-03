# Third-party components

The MIT license at the repository root covers the team's source contributions. Installed dependencies, downloaded base models, and separately obtained challenge data remain subject to their respective terms. No base weights, fine-tuned weights, learned artifact kit, or challenge data are included in this source release.

The historical pipeline used the following base models. These revisions and license descriptions are recorded in the final submission package; they are not a new license audit of downloaded files.

- Dense bi-encoder: `intfloat/multilingual-e5-small`, approximately 117.7M parameters, recorded MIT license; revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- Cross-encoder feature: `intfloat/multilingual-e5-base`, approximately 278M parameters, recorded MIT license; revision `d128750597153bb5987e10b1c3493a34e5a4502a`.
- Pair classifier: `Qwen/Qwen2.5-7B`, approximately 7.6B parameters, recorded Apache-2.0 license; revision `d149729398750b98c0af14eb82c78cfe92750796`.
- Second pair classifier: `Qwen/Qwen2.5-1.5B`, approximately 1.5B parameters, recorded Apache-2.0 license; revision `8faed761d45a263340a0528343f099c05c9a4323`.

LightGBM supplies the tabular matchers and combiners. Other dependencies are listed in the two pipeline requirements files. In particular, Unidecode has its own GPL license; the MIT license here does not change that dependency's license. Consult the upstream packages and model files when using or redistributing their components.
