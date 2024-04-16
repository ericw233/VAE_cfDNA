import torch
import torch.nn as nn
from reverse_layer import ReverseLayerF


class DANN_1D(nn.Module):
    def __init__(self, input_size, num_class, num_domain, out1, out2, conv1, pool1, drop1, conv2, pool2, drop2, fc1, fc2, drop3):
        super(DANN_1D, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=out1, kernel_size=conv1, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(out1),
            nn.Dropout(drop1),
            nn.MaxPool1d(kernel_size=pool1, stride=2),
            
            nn.Conv1d(in_channels=out1, out_channels=out2, kernel_size=conv2, stride=2, bias=None),
            nn.ReLU(),
            nn.BatchNorm1d(out2),
            nn.Dropout(drop2),
            nn.MaxPool1d(kernel_size=pool2, stride=2)
        )
        
        self.fc_input_size = self._get_fc_input_size(input_size)
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.r01b_classifier = nn.Sequential(
            nn.Linear(self.fc_input_size, fc1),
            nn.ReLU(),
            nn.Dropout(drop3),
            nn.Linear(fc1, fc2),
            nn.Linear(fc2, 1),
            nn.Sigmoid()
        )
     
        
    def _get_fc_input_size(self, input_size):
        dummy_input = torch.randn(1, 1, input_size)
        x = self.feature_extractor(dummy_input)
        flattened_size = x.size(1) * x.size(2)
        return flattened_size
    
    @staticmethod
    def initialize_lin(layer, bias=0):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.constant_(layer.bias, bias)
                
    def forward(self, x, y, alpha):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        
        # task classifier output
        task_output = self.task_classifier(features)
   
        # domain classifier output
        reverse_features = ReverseLayerF.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)
        
        if y is not None:
            feature_r01b = self.feature_extractor(y)
            feature_r01b = feature_r01b.view(feature_r01b.size(0), -1)
            r01b_output = self.r01b_classifier(feature_r01b)
            
            return task_output.squeeze(1), domain_output.squeeze(1), r01b_output.squeeze(1)
        else:
            return task_output.squeeze(1), domain_output.squeeze(1), None
    
    
class VAE(nn.Module):
    def __init__(self, input_size, code_size=64):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.code_size = code_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.code_mean = nn.Linear(256, code_size)
        self.code_sd = nn.Linear(256, code_size)

        self.decoder = nn.Sequential(
            nn.Linear(code_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_size),
            nn.Sigmoid()
        )
        
        # Task classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(code_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(code_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Task classifier at the reconstructed layer
        self.task_classifier2 = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,1),
            nn.Sigmoid()
        )
        
    def forward(self, x, alpha=0.01):
        encoding = self.encoder(x)
        code_mean = self.code_mean(encoding)
        code_sd = self.code_sd(encoding)
        noise = torch.randn_like(code_sd)
        code = code_mean + torch.exp(0.5 * code_sd) * noise
               
        code_task = self.task_classifier(code)
        reverse_code = ReverseLayerF.apply(code,alpha)        
        code_domain = self.domain_classifier(reverse_code)
        
        decoding = self.decoder(code)
        output_task = self.task_classifier2(decoding)
        
        return decoding, code_mean, code_sd, code, code_task.squeeze(1), code_domain.squeeze(1), output_task.squeeze(1)
    
    