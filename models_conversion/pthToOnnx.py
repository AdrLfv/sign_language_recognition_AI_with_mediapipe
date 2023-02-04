import numpy as np
import onnxruntime
import torch
from training.model import myLSTM
from os import path


def export_to_onnx(input_size, hidden_size, num_layers, output_size, device, DIR_PATH):
    """ Exports the pytorch model to onnx """
    model = myLSTM(input_size, hidden_size, num_layers, output_size, device)

    WEIGHTS_PATH = path.join(DIR_PATH, "outputs/slr_"+str(output_size)+".pth")
    OUTPUT_PATH = path.join(DIR_PATH, "outputs/slr_"+str(output_size)+".onnx")
    model.load_state_dict(torch.load(WEIGHTS_PATH))
    model.eval()
    input = torch.randn(1, 30, input_size, requires_grad=True, device=device)
    torch_out = model(input)

    torch.onnx.export(
        model,
        input,
        OUTPUT_PATH,
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names = ['input'],
        output_names = ['output'],
        dynamic_axes={'input' : {0 : 'batch_size'},
                    'output' : {0 : 'batch_size'}})

    # ort_session = onnxruntime.InferenceSession(OUTPUT_PATH,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_session = onnxruntime.InferenceSession(OUTPUT_PATH,providers=['CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")