BENCHMARK = (
    "B0", "B0_MATCHED", "B0_EARLY", "B0_MIDDLE", "B0_ALLTOKENS", "GENERIC_R46",
    "GLOBAL_RECT", "GLOBAL_LINEAR", "BIDIR_PAIR", "ALIGN_SHUFFLE", "LP",
    "HOULSBY", "ADAPTFORMER", "ADAPTERPLUS", "ADAPTERDROP",
)
MATCHED = (
    "LINEAR_REF", "LINEAR_SHUFFLED", "LINEAR_UNIFORM_REFERENCE", "LINEAR_NOAUX",
    "LINEAR_POOL_ONLY", "BASE_MATCHED",
)
DATASETS = ("brset", "deepdrid", "mbrset", "messidor2", "sustech", "paraguay", "aptos")
SEEDS = (823, 289, 592, 605, 329)
FOLDS = (1, 2, 3)


def cache_name(phase, model):
    if phase == "matched":
        if model not in MATCHED:
            raise ValueError("Unknown matched control")
        return "pc.npy"
    if model not in BENCHMARK:
        raise ValueError("Unknown benchmark model")
    if model == "LP":
        return "features.npy"
    if model in ("HOULSBY", "ADAPTFORMER", "ADAPTERPLUS"):
        return "prefix0.npy"
    if model == "ADAPTERDROP":
        return "prefix11.npy"
    if model in ("B0_MATCHED", "B0_ALLTOKENS"):
        return "prefix10.npy"
    if model == "B0_EARLY":
        return "prefix2.npy"
    if model == "B0_MIDDLE":
        return "prefix6.npy"
    return "pc.npy"


def is_published_kernel(model):
    return model in ("B0_MATCHED", "B0_EARLY", "B0_MIDDLE", "B0_ALLTOKENS", "LP", "HOULSBY", "ADAPTFORMER", "ADAPTERPLUS", "ADAPTERDROP")


def make_model(phase, model, seed, encoder, device):
    if phase == "matched":
        if model == "BASE_MATCHED":
            from lesion_base_matched_v1.model import make_model as create
        elif model == "LINEAR_POOL_ONLY":
            from lesion_pc_linear_pooling_v1.model import make_model as create
        else:
            from lesion_pc_linear_ablation_v1.model import make_model as create
        value = create(model, seed=seed, encoder=encoder, device=device)
    elif model == "B0":
        from lesion_direction.model import DirectionModel
        value = DirectionModel(encoder, arm="B0", seed=seed)
    elif model in ("GLOBAL_RECT", "GLOBAL_LINEAR"):
        from lesion_global.model import GlobalPairContrastModel
        value = GlobalPairContrastModel(encoder, variant=model, seed=seed)
    elif model == "BIDIR_PAIR":
        from lesion_bidirectional.model import BidirectionalPairModel
        value = BidirectionalPairModel(encoder, seed=seed)
    elif model == "GENERIC_R46":
        from lesion_bidirectional.model import GenericRank46Model
        value = GenericRank46Model(encoder, seed=seed)
    elif model == "ALIGN_SHUFFLE":
        from lesion_alignment.model import AlignmentModel
        value = AlignmentModel(encoder, variant=model, seed=seed)
    elif model in ("LP", "HOULSBY", "ADAPTFORMER"):
        from lesion_published_baselines.model import PublishedBaseline
        value = PublishedBaseline(encoder, arm=model, seed=seed)
    elif model in BENCHMARK:
        from lesion_close_adapters.model import CloseAdapterModel
        from lesion_close_adapters.model import ARM_SPLIT_BLOCK
        start = ARM_SPLIT_BLOCK[model]
        value = CloseAdapterModel(encoder, arm=model, seed=seed, input_split_block=start)
    else:
        raise ValueError("Unknown experiment model")
    value.run_seed = seed
    return value.to(device)
