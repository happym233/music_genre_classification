import torch


def test_MLP():
    from models.MLP import MLP
    model = MLP(10, 10, [50])
    inp_tensor = torch.rand([10, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 10])


def test_CNN_1d_block():
    from models.CNN_block import CNN_1d_block
    model = CNN_1d_block(in_channels=1,
                         out_channels=4,
                         kernel_size=3,
                         stride=1,
                         padding=0,
                         pooling='avg',
                         pooling_kernel_size=2,
                         pooling_stride=1,
                         pooling_padding=0,
                         activation='ReLU',
                         batch_norm=True)
    inp_tensor = torch.rand([10, 1, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 4, 7])


def test_CNN_2d_block():
    from models.CNN_block import CNN_2d_block
    model = CNN_2d_block(in_channels=1,
                         out_channels=4,
                         kernel_size=3,
                         stride=1,
                         padding=0,
                         pooling='avg',
                         pooling_kernel_size=2,
                         pooling_stride=1,
                         pooling_padding=0,
                         activation='ReLU',
                         batch_norm=True)
    inp_tensor = torch.rand([10, 1, 10, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 4, 7, 7])


def test_Res1d():
    from models.CNN_block import Res1d
    model = Res1d(2)
    inp_tensor = torch.rand([10, 2, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 2, 10])


def test_Res2d():
    from models.CNN_block import Res2d
    model = Res2d(2)
    inp_tensor = torch.rand([10, 2, 10, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 2, 10, 10])


def test_MusicCNN1d():
    from models.MusicCNN1d import MusicCNN1d
    model = MusicCNN1d(CNN_input_channels=1,
                       CNN_out_channels=[2, 4, 8],
                       pooling='avg',
                       DNN_input_dim=8,
                       DNN_output_dim=10,
                       DNN_hidden_dims=[50],
                       res_block=False)
    inp_tensor = torch.rand([10, 1, 100])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 10])


def test_MusicCNN2d():
    from models.MusicCNN2d import MusicCNN2d
    model = MusicCNN2d(out_channels=[2, 4, 8], DNN_input_dim=1152, DNN_hidden_dims=[], DNN_output_dim=10,
                       res_block=False)
    inp_tensor = torch.rand([10, 1, 100, 100])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 10])


def test_LSTM():
    from models.MusicLSTM import MusicLSTM
    model = MusicLSTM(input_size=10, hidden_size=20, output_size=10, num_layers=4, dropout=0.15, bidirectional=False)
    inp_tensor = torch.rand([10, 10, 10])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 10])


def test_CRDNN():
    from models.MusicCRDNN import MusicCRDNN
    model = MusicCRDNN(CNN_out_channels=[2, 4, 8, 8],
                       output_dim=10,
                       LSTM_input_size=48,
                       LSTM_hidden_size=80,
                       LSTM_num_layers=8,
                       MLP_hidden_dims=[160],
                       res_block=False,
                       bidirectional=False)
    inp_tensor = torch.rand([10, 1, 100, 100])
    out_tensor = model(inp_tensor)
    assert out_tensor.shape == torch.Size([10, 10])
