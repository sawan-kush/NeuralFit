from helpers import make_dataset_zip, make_state_dict, to_bytes, upload_session


def test_supported_architectures(client):
    data = client.get("/api/v1/architectures").json()
    assert {a["key"] for a in data} == {"resnet18", "mobilenet_v2"}
    resnet = next(a for a in data if a["key"] == "resnet18")
    assert resnet["default_input_size"] == 224 and resnet["classifier_weight_key"] == "fc.weight"


def test_methods_list_only_implemented_methods(client):
    methods = {m["key"]: m for m in client.get("/api/v1/methods").json()}
    assert set(methods) == {"fp16", "int8_ptq"}  # pruning must not be offered until implemented
    assert methods["int8_ptq"]["available"] is True
    backend = next(o for o in methods["int8_ptq"]["options"] if o["name"] == "backend")
    assert "fbgemm" in backend["choices"]
    assert all(m["limitations"] for m in methods.values())


def test_unsupported_architecture_rejected(client):
    r = upload_session(client, architecture="vgg16", weights=to_bytes(make_state_dict()), dataset=make_dataset_zip())
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "unsupported_architecture"


def test_unknown_route_uses_error_shape(client):
    r = client.get("/api/v1/nope")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
