@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = load_backbone(config)

        self.decode_head = UperNetHead(config, in_channels=self.backbone.channels)
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None        
                
        self.unet1_bl1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),                                      
            nn.ReLU(inplace=True),                                   
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),                                      
            nn.ReLU(inplace=True),                                   
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.unet1_bl2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),                                      
            nn.ReLU(inplace=True),                                   
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False), 
            nn.BatchNorm2d(64),                                      
            nn.ReLU(inplace=True),                                   
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.unet2_bl1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(128),                                        
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(128),                                        
            nn.ReLU(inplace=True),                                      
        )
        
        self.unet2_bl2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(128),                                        
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.BatchNorm2d(128),                                        
            nn.ReLU(inplace=True),                                      
        )
        
        self.up1_bl1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                       
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True)                                       
        )
        
        self.up1_bl2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                       
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True)                                       
        )
        
        self.up2_bl1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
        )
        
        self.up2_bl2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
        )
        
        self.unet3_bl1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
        )
        
        self.unet3_bl2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                      
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),  
        )
        
        self.final_bl1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                    
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                    
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
        )
        
        self.final_bl2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                    
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),  
            nn.ReLU(inplace=True),                                    
            nn.Conv2d(64, 1, kernel_size=3, padding=1, bias=False),
        )
        
        self.post_init()

    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SemanticSegmenterOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, UperNetForSemanticSegmentation
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> image_processor = AutoImageProcessor.from_pretrained("openmmlab/upernet-convnext-tiny")
        >>> model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-convnext-tiny")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/fixtures_ade20k", filename="ADE_val_00000001.jpg", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")

        >>> inputs = image_processor(images=image, return_tensors="pt")

        >>> outputs = model(**inputs)

        >>> logits = outputs.logits  # shape (batch_size, num_labels, height, width)
        >>> list(logits.shape)
        [1, 150, 512, 512]
        ```"""
        if labels is not None and self.config.num_labels == 1:
            raise ValueError("The number of labels should be greater than one")

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
        features = outputs.feature_maps
        
        logits = self.decode_head(features)
        
        unet_out_1_bl1 = self.unet1_bl1(pixel_values)
        unet_out_1_bl2 = self.unet1_bl2(pixel_values)
        
        unet_out_2_bl1 = self.unet2_bl1(unet_out_1_bl1)
        unet_out_2_bl2 = self.unet2_bl2(unet_out_1_bl2)
        
        logits_up_1_bl1 = self.up1_bl1(logits[:, 0:1, :, :])
        logits_up_1_bl2 = self.up1_bl2(logits[:, 1:2, :, :])
        
        logits_cat_1_bl1 = torch.cat((logits_up_1_bl1, unet_out_2_bl1), 1)
        logits_cat_1_bl2 = torch.cat((logits_up_1_bl2, unet_out_2_bl2), 1)
        
        logits_up_2_bl1 = self.up2_bl1(logits_up_1_bl1)
        logits_up_2_bl2 = self.up2_bl2(logits_up_1_bl2)
        
        unet_out_3_bl1 = self.unet3_bl1(logits_cat_1_bl1)
        unet_out_3_bl2 = self.unet3_bl2(logits_cat_1_bl2)
        
        logits_cat_2_bl1 = torch.cat((logits_up_2_bl1, unet_out_3_bl1), 1)
        logits_cat_2_bl2 = torch.cat((logits_up_2_bl2, unet_out_3_bl2), 1)
        
        logits_bl1 = self.final_bl1(logits_cat_2_bl1)
        logits_bl2 = self.final_bl2(logits_cat_2_bl2)  
        
        logits = torch.cat((logits_bl1, logits_bl2), 1) 
        
        loss_bl1 = None
        loss_bl2 = None
        loss = None
        
        if labels is not None:
            loss_fct_bl1 = nn.MSELoss()
            loss_fct_bl2 = nn.BCEWithLogitsLoss()  
            
            label1 = labels[:, 0:1, :, :]  
            label2 = labels[:, 1:2, :, :]  
            
            valid_mask = (label1 != -1).float()
            valid_samples = valid_mask.view(label1.size(0), -1).sum(dim=1) > 0  
            
            if valid_samples.any():
                logits_bl1_valid = logits_bl1[valid_samples]
                label1_valid = label1[valid_samples]
                
                loss_bl1 = loss_fct_bl1(logits_bl1_valid, label1_valid)
            else:
                loss_bl1 = logits_bl1.sum() * 0.0
            
            loss_bl2 = loss_fct_bl2(logits_bl2, label2.float())
            
            loss = 0.05 * loss_bl1 + 2.0 * loss_bl2
            
                    
        if not return_dict:
            if output_hidden_states:
                output = (logits,) + outputs[1:]
            else:
                output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SemanticSegmenterOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )