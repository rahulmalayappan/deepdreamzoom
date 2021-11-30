class UnsupportedShapeError(Exception):
    def __init__(self, message, shape):
        self.shape = shape
        super().__init__(message)

def get_loss_fn(model, loss_weights):
    loss_value = 0.
    
    def make_hook(weight, channels):
        def hook(model, input_tensor, output_tensor):
            nonlocal loss_value
            L = output_tensor.clone() # Clone to prevent issues with in-place ReLU
            if channels is not None:
                if len(L.shape) == 4:
                    L = L[:, channels, :, :]
                elif len(L.shape) == 2:
                    L = L[:, channels]
                else:
                    raise UnsupportedShapeError(
                        "Unsupported shape of activations for channel slicing: {}".format(L.shape),
                        L.shape
                    )
                
            loss_value += (L * L).mean() * weight
        return hook
        
    handles = []
    for layername, weight, channels in loss_weights:
        module = model
        if type(layername) is str:
            module = dict(module.named_children())[layername]
        else:
            for l in layername:
                module = dict(module.named_children())[l]
                
        handles.append(module.register_forward_hook(make_hook(weight, channels)))
        
    def loss_fn(img):
        nonlocal model, loss_value
        loss_value = 0.
        model(img) # Hooks will populate loss_value
        return loss_value
    
    return loss_fn, handles
