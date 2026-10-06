import onnxruntime as ort
import torch

from bloomer.predict import load_model


def main() -> None:
    model = load_model("best_model.pt")

    example_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        (example_input,),
        "bloomer.onnx",
        input_names=["input"],
        output_names=["logits"],
        external_data=False,
    )

    print("Exported to bloomer.onnx")

    session = ort.InferenceSession("bloomer.onnx")
    test_input = torch.randn(1, 3, 224, 224)

    onnx_output = session.run(None, {"input": test_input.numpy()})[0]
    with torch.no_grad():
        torch_output = model(test_input).numpy()

    print("Largest difference:", abs(onnx_output - torch_output).max())
