import numpy as np
import onnxruntime
import torch.onnx
from ..training.LSTM import myLSTM


def export_to_onnx(output_size):
    """ Exports the pytorch model to onnx """
    model = myLSTM(150, 128, 2, output_size)
    WEiGHTS_PATH = "./models/slr_"+str(output_size)+".pth"
    OUTPUT_PATH = "./models/slr_"+str(output_size)+".onnx"
    model.load_state_dict(torch.load(WEiGHTS_PATH))
    model.eval()
    input = torch.randn(1, 30, 150, requires_grad=True)
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

    ort_session = onnxruntime.InferenceSession(OUTPUT_PATH,providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
    ort_outs = ort_session.run(None, ort_inputs)

    print(np.argmax(ort_outs[-1]))
    print(torch.argmax(torch_out.squeeze(0)))
    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")