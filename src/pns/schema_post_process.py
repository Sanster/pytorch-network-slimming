def mbv3_large_schema_post_process(config):
    # BN in block with se module should have same channels
    se_shortcuts = []
    modules = []
    for m in config["modules"]:
        if m["name"].endswith("fc2"):
            # e.g: features.5.block
            feature_name = ".".join(m["name"].split(".")[:-2])
            m["next_bn"] = f"{feature_name}.1.1"
        elif m["name"].endswith("fc1"):
            m["next_bn"] = ""

        modules.append(m)
    config["modules"] = modules
