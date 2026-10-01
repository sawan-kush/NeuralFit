"""Upload validation: weights, dataset structure, archive safety, limits and error hygiene."""
import io
import json
import zipfile

import torch

from helpers import (
    LABELS, make_dataset_zip, make_image_bytes, make_state_dict, session_config, to_bytes, upload_session,
)
from app.models.registry import get_arch


def _ok_weights(arch="resnet18"):
    return to_bytes(make_state_dict(arch))


def _err(resp, status, code):
    assert resp.status_code == status, resp.text
    assert resp.json()["error"]["code"] == code
    return resp.json()["error"]["message"]


def _no_session_left(settings):
    assert list(settings.sessions_dir.iterdir()) == []


# ------------------------------------------------------------------ weights
def test_valid_session_summary(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip())
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["num_classes"] == 3 and body["total_images"] == 12
    assert body["class_counts"] == {"cats": 4, "dogs": 4, "birds": 4}
    assert body["preprocessing"]["input_size"] == 64
    assert client.get(f"/api/v1/sessions/{body['session_id']}").status_code == 200


def test_wrapper_folder_is_accepted(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(wrapper="validation_dataset"))
    assert r.status_code == 201 and r.json()["total_images"] == 12


def test_mobilenet_weights_accepted(client):
    r = upload_session(client, architecture="mobilenet_v2", weights=_ok_weights("mobilenet_v2"),
                       dataset=make_dataset_zip())
    assert r.status_code == 201, r.text


def test_wrong_weights_extension(client, settings):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(), weights_name="weights.txt")
    _err(r, 400, "invalid_upload")
    _no_session_left(settings)


def test_garbage_weights(client, settings):
    r = upload_session(client, weights=b"this is not a checkpoint", dataset=make_dataset_zip())
    _err(r, 422, "invalid_weights")
    _no_session_left(settings)


def test_full_pickled_model_is_rejected(client):
    model = get_arch("resnet18").build_float(3)
    r = upload_session(client, weights=to_bytes(model), dataset=make_dataset_zip())  # torch.save(model), not state dict
    msg = _err(r, 422, "invalid_weights")
    assert "state_dict" in msg


def test_weights_for_wrong_architecture(client):
    r = upload_session(client, architecture="resnet18", weights=_ok_weights("mobilenet_v2"), dataset=make_dataset_zip())
    msg = _err(r, 422, "invalid_weights")
    assert "do not look like" in msg


def test_label_count_must_match_model_outputs(client):
    labels = ["a", "b", "c", "d"]
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(labels),
                       config=session_config(labels))
    msg = _err(r, 422, "invalid_weights")
    assert "3 output classes" in msg and "4 class labels" in msg


def test_state_dict_with_missing_keys(client):
    sd = make_state_dict()
    sd.pop("layer1.0.conv1.weight")
    r = upload_session(client, weights=to_bytes(sd), dataset=make_dataset_zip())
    msg = _err(r, 422, "invalid_weights")
    assert "missing" in msg


def test_wrapped_and_dataparallel_state_dicts_are_accepted(client):
    sd = make_state_dict()
    wrapped = {"state_dict": {f"module.{k}": v for k, v in sd.items()}, "epoch": 3}
    r = upload_session(client, weights=to_bytes(wrapped), dataset=make_dataset_zip())
    assert r.status_code == 201, r.text


def test_weights_size_limit(make_client):
    c = make_client(max_weights_bytes=1000)
    r = upload_session(c, weights=_ok_weights(), dataset=make_dataset_zip())
    _err(r, 413, "payload_too_large")


def test_request_size_limit_middleware(make_client):
    c = make_client(max_weights_bytes=10, max_dataset_zip_bytes=10)
    r = c.post("/api/v1/sessions", content=b"x" * (3 * 1024**2), headers={"content-type": "multipart/form-data"})
    _err(r, 413, "payload_too_large")


# ------------------------------------------------------------------ config
def test_invalid_config_json(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(), config="{not json")
    _err(r, 400, "invalid_config")


def test_duplicate_and_short_labels(client):
    for labels in (["a", "a"], ["only-one"]):
        r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(),
                           config=json.dumps({"class_labels": labels}))
        _err(r, 400, "invalid_config")


# ------------------------------------------------------------------ dataset
def test_empty_dataset(client, settings):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("readme.txt", "no images here")
    r = upload_session(client, weights=_ok_weights(), dataset=buf.getvalue())
    _err(r, 422, "empty_dataset")
    _no_session_left(settings)


def test_not_a_zip(client):
    r = upload_session(client, weights=_ok_weights(), dataset=b"plain text, not a zip")
    _err(r, 422, "invalid_archive")


def test_wrong_dataset_extension(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(), dataset_name="dataset.tar")
    _err(r, 400, "invalid_upload")


def test_folder_not_in_labels(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(["cats", "dogs", "fish"]))
    msg = _err(r, 422, "label_mismatch")
    assert "fish" in msg


def test_images_directly_in_root(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("loose.jpg", make_image_bytes(0, 0))
    r = upload_session(client, weights=_ok_weights(), dataset=buf.getvalue())
    _err(r, 422, "invalid_structure")


def test_missing_class_folder_is_a_warning(client):
    r = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip(["cats", "dogs"]))
    assert r.status_code == 201
    assert any("birds" in w for w in r.json()["warnings"])
    assert r.json()["class_counts"]["birds"] == 0


def test_non_image_files_are_ignored_with_warning(client):
    zip_bytes = make_dataset_zip(extra={"notes.txt": b"hello", "cats/.DS_Store": b"junk"})
    r = upload_session(client, weights=_ok_weights(), dataset=zip_bytes)
    assert r.status_code == 201
    assert r.json()["total_images"] == 12
    assert any("ignored" in w for w in r.json()["warnings"])


def test_corrupt_image_rejected(client):
    zip_bytes = make_dataset_zip(extra={"cats/broken.jpg": b"\xff\xd8 definitely not a jpeg"})
    r = upload_session(client, weights=_ok_weights(), dataset=zip_bytes)
    msg = _err(r, 422, "invalid_image")
    assert "broken.jpg" in msg


def test_too_many_images(make_client):
    c = make_client(max_dataset_images=5)
    r = upload_session(c, weights=_ok_weights(), dataset=make_dataset_zip())
    _err(r, 422, "archive_too_large")


# ------------------------------------------------------------------ archive safety
def _zip_with(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item, data in entries:
            zf.writestr(item, data)
    return buf.getvalue()


def test_zip_slip_is_rejected_and_nothing_escapes(client, settings):
    evil = _zip_with([("../evil.jpg", make_image_bytes(0, 0)), ("cats/a.jpg", make_image_bytes(0, 1))])
    r = upload_session(client, weights=_ok_weights(), dataset=evil)
    _err(r, 422, "unsafe_archive")
    assert not (settings.data_dir / "evil.jpg").exists()
    assert not (settings.sessions_dir.parent / "evil.jpg").exists()
    _no_session_left(settings)


def test_absolute_and_backslash_paths_rejected(client):
    for name in ("/etc/cats/a.jpg", "cats\\..\\..\\a.jpg", "C:/cats/a.jpg"):
        r = upload_session(client, weights=_ok_weights(), dataset=_zip_with([(name, make_image_bytes(0, 0))]))
        _err(r, 422, "unsafe_archive")


def test_symlink_entries_rejected(client):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("cats/link.jpg")
        info.external_attr = 0o120777 << 16
        zf.writestr(info, "/etc/passwd")
    r = upload_session(client, weights=_ok_weights(), dataset=buf.getvalue())
    _err(r, 422, "unsafe_archive")


def test_zip_bomb_ratio_rejected(client):
    bomb = _zip_with([("cats/big.jpg", b"\x00" * (8 * 1024 * 1024))])
    r = upload_session(client, weights=_ok_weights(), dataset=bomb)
    _err(r, 422, "unsafe_archive")


def test_extracted_size_limit(make_client):
    c = make_client(max_extracted_bytes=2000)
    r = upload_session(c, weights=_ok_weights(), dataset=make_dataset_zip())
    _err(r, 422, "archive_too_large")


# ------------------------------------------------------------------ hygiene
def test_error_responses_never_leak_local_paths(client, settings):
    responses = [
        upload_session(client, weights=b"garbage", dataset=make_dataset_zip()),
        upload_session(client, weights=_ok_weights(), dataset=b"not zip"),
        upload_session(client, weights=_ok_weights(), dataset=_zip_with([("../x.jpg", b"1")])),
        client.get("/api/v1/sessions/" + "0" * 32),
        client.get("/api/v1/runs/" + "0" * 32),
    ]
    for r in responses:
        assert r.status_code >= 400
        assert str(settings.data_dir) not in r.text
        assert set(r.json()) == {"error"} and set(r.json()["error"]) == {"code", "message"}


def test_delete_session(client, settings):
    sid = upload_session(client, weights=_ok_weights(), dataset=make_dataset_zip()).json()["session_id"]
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert client.get(f"/api/v1/sessions/{sid}").status_code == 404
    assert client.delete(f"/api/v1/sessions/{sid}").status_code == 404
    _no_session_left(settings)


def test_session_id_path_traversal_is_a_plain_404(client):
    assert client.get("/api/v1/sessions/..%2F..%2Fetc").status_code == 404
