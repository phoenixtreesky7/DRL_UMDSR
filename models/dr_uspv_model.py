import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import dr_uspv_net
from epdn import epdn_networks
import math
#from ECLoss.ECLoss import DCLoss
#from TVLoss.TVLossL1 import TVLossL1 
#from models.gradient import gradient
from torchvision import transforms

class DR_USPV_MODEL(BaseModel):
    def name(self):
        return 'DR_USPV_MODEL'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.flip = transforms.RandomHorizontalFlip(p=1)

        self.gpu_ids
        self.hl = opt.hl
        self.opt = opt
        self.dh_real = opt.dh_real
        self.allmodel = opt.allmodel

        # specify the training losses you want to print out. The program will call base_model.get_current_losses  loss_style_B + self.loss_content_A + self.loss_content_B
        self.loss_names = ['D_A', 'G_A', 'idt_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'style_B']
        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = ['real_A', 'fake_B', 'idt_A', 'rec_A']
        visual_names_B = ['real_B', 'fake_A','idt_B','rec_B'] # , 'fake_PA', 'rec_PA',, 'cycle_PA', 'G_PA'
        visual_names_C = ['gradient_real_A','gradient_fake_B','gradient_real_B','gradient_fake_A']
        #if self.isTrain and self.opt.lambda_identity > 0.0:
        #    visual_names_A.append(['idt_A', 'idt_A_fine'])
        #    visual_names_B.append(['idt_B'])

        if self.isTrain:
            self.visual_names = visual_names_A + visual_names_B #+ visual_names_C
        else:
            self.visual_names = visual_names_A + visual_names_B
        # specify the models you want to save to the disk. The program will call base_model.save_networks and base_model.load_networks
        if self.isTrain:
            self.model_names = ['G_E_A', 'G_E_B', 'G_D_A', 'G_D_B', 'D_A', 'D_B']

        else:  # during test time, only load Gs
            self.model_names = ['G_E_A', 'G_E_B', 'G_D_A', 'G_D_B']

        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        self.netG_E_A = dr_uspv_net.define_G_E_A(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.n_blocks, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_D_A = dr_uspv_net.define_G_D_A(opt.input_nc, opt.output_nc,
                                        opt.ngf, opt.n_blocks, opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_E_B = dr_uspv_net.define_G_E_B(opt.input_nc, opt.input_nc,
                                        opt.ngf, opt.n_blocks, opt.hl, opt.which_model_netG, opt.ca_type, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_D_B = dr_uspv_net.define_G_D_B(opt.input_nc, opt.input_nc,
                                        opt.ngf, opt.n_blocks, opt.hl, opt.which_model_netG, opt.fuse_model, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            if self.opt.which_model_netD == 'fd' or self.opt.which_model_netD == 'nl':
                self.netD_A = dr_uspv_net.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_B = dr_uspv_net.define_D(opt.input_nc, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            else:
                self.netD_A = dr_uspv_net.define_D(opt.input_nc*2, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
                self.netD_B = dr_uspv_net.define_D(opt.input_nc*2, opt.ndf,
                                                opt.which_model_netD,
                                                opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.isTrain:
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.fake_PA_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = dr_uspv_net.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            self.criterionVGG = epdn_networks.VGGLoss(self.gpu_ids)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_E_A.parameters(), self.netG_E_B.parameters(), self.netG_D_A.parameters(), self.netG_D_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device) # real world clear image
        self.real_B = input['B' if AtoB else 'A'].to(self.device) # real world hazy image
        #self.real_C = input['C'].to(self.device) # real world hazy image
        #self.real_D = input['D'].to(self.device) # real world hazy image
        #self.real_C = input['C'].to(self.device)
        self.image_paths = input['B_paths' if AtoB else 'B_paths']
        


    def forward(self):
        # Forward Cycle #
        # disentanglement #
        if self.allmodel:
            self.content_A = self.netG_E_A(self.real_A)
            self.content_B, self.style_B = self.netG_E_B(self.real_B)


            # translation #
            self.fake_B = self.netG_D_B(self.real_A, self.content_A, self.style_B)
            self.fake_A = self.netG_D_A(self.real_B, self.content_B)  # , self.fake_PA

            # identity #
            self.idt_A = self.netG_D_A(self.real_A, self.content_A)  # , self.idt_PA
            self.idt_B = self.netG_D_B(self.real_B, self.content_B, self.style_B)

            # Backward Cycle #
            # disentanglement #
            self.content_fake_A = self.netG_E_A(self.fake_A)
            self.content_fake_B, self.style_fake_B = self.netG_E_B(self.fake_B)

            # translation #
            self.rec_A = self.netG_D_A(self.real_A, self.content_fake_B) # , self.rec_PA
            self.rec_B = self.netG_D_B(self.real_B, self.content_fake_A, self.style_fake_B)

        else:
            self.content_B, self.style_B = self.netG_E_B(self.real_B)

    def backward_FD_basic(self, netD, ref, real, fake):
        # Real
        pred_real = netD(ref, real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(ref.detach(), fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_FD_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_FD_basic(self.netD_B, self.flip(self.real_C), self.real_A, fake_A)


    def backward_FD_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_FD_basic(self.netD_A, self.flip(self.real_D), self.real_B, fake_B)

    def backward_CD_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_CD_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        dehaze_real_B_real_A_cat = torch.cat((self.real_B, self.real_A), dim=1)
        dehaze_real_B_fake_A_cat = torch.cat((self.real_B, fake_A), dim=1)
        self.loss_D_B = self.backward_D_basic(self.netD_B, dehaze_real_B_real_A_cat, dehaze_real_B_fake_A_cat)

    def backward_CD_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        dehaze_real_A_real_B_cat = torch.cat((self.real_A, self.real_B), dim=1)
        dehaze_real_A_fake_B_cat = torch.cat((self.real_A, fake_B), dim=1)
        self.loss_D_A = self.backward_D_basic(self.netD_A, dehaze_real_A_real_B_cat, dehaze_real_A_fake_B_cat)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_A, fake_A)

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_B, fake_B)


    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_C = self.opt.lambda_content
        #lambda_S = self.opt.lambda_style
        
        # Identity loss
        if lambda_idt > 0:
            # G_D_A should be identity if real_A is fed.
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A) * lambda_B * lambda_idt #* 0.382
            # G_D_B should be identity if real_B is fed.
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B) * lambda_A * lambda_idt 
            # G_D_A should be identity if real_A is fed.
            #self.loss_idt_PA = self.criterionIdt(self.idt_PA, self.real_A) * lambda_A * lambda_idt * 0.618

        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0
            #self.loss_idt_A_fine = 0

        if self.opt.which_model_netD == 'fd':
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.flip(self.real_D), self.fake_B), True) * 0.16
            #* 2#* 0.382
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.flip(self.real_C), self.fake_A), True) * 0.16
            # GAN loss D_A(G_A(A))
            #self.loss_G_PA = self.criterionGAN(self.netD_B(self.real_D, self.fake_PA), True)
        elif self.opt.which_model_netD == 'cd':
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(torch.cat((self.real_A, self.fake_B), dim=1)), True) * 0.16
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(torch.cat((self.real_B, self.fake_A), dim=1)), True) * 0.16
        elif self.opt.which_model_netD == 'nl':
            # GAN loss D_A(G_A(A))
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True) * 1
            #* 0.382
            # GAN loss D_B(G_B(B))
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True) * 1
            # GAN loss D_A(G_A(A))
            #self.loss_G_B_fine = self.criterionGAN(self.netD_B(torch.cat((self.real_B, self.fake_A_fine), dim=1)), True)* 0.618
         
        # Forward cycle loss
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A * 3 
        # Backward cycle loss
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B * 3
        # Forward cycle loss
        #self.loss_cycle_PA = self.criterionCycle(self.rec_PA, self.real_A) * lambda_A * 3 
        
        # combined loss
        self.loss_G = self.loss_idt_A + self.loss_idt_B + self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B# + self.loss_cycle_PA + self.loss_G_PA #+ self.loss_idt_PA 


        # Vgg loss
        self.loss_G_VGG_recA = self.criterionVGG(self.rec_A, self.real_A) * self.opt.lambda_vgg * 2#* 0.382
        self.loss_G_VGG_recB = self.criterionVGG(self.rec_B, self.real_B) * self.opt.lambda_vgg * 2
        #self.loss_G_VGG_recA_fine = self.criterionVGG(self.rec_A_fine, self.real_A) * self.opt.lambda_vgg * 0.618
        self.loss_G += self.loss_G_VGG_recA + self.loss_G_VGG_recB #+ self.loss_G_VGG_recA_fine

        # Gradient loss
        #self.gradient_real_A = gradient(self.real_A)
        #self.gradient_real_B = gradient(self.real_B)
        #self.gradient_fake_B = gradient(self.fake_B)
        #self.gradient_fake_A = gradient(self.fake_A)
        #self.gradient_rec_A = gradient(self.rec_A)
        #self.gradient_rec_B = gradient(self.rec_B)

        #self.loss_gradient_fake_B = 0.2 * self.criterionIdt(self.gradient_real_A, self.gradient_fake_B)
        #self.loss_gradient_fake_A = 0.2 * self.criterionIdt(self.gradient_real_B, self.gradient_fake_A)
        #self.loss_gradient_rec_A = 0.2 * self.criterionIdt(self.gradient_real_A, self.gradient_rec_A)
        #self.loss_gradient_rec_B = 0.2 * self.criterionIdt(self.gradient_real_B, self.gradient_rec_B)

        #self.loss_G += self.loss_gradient_fake_B + self.loss_gradient_fake_A + self.loss_gradient_rec_A + self.loss_gradient_rec_B

        # Content Loss
        if self.hl > 0:
            self.loss_content_A = self.criterionIdt(self.content_A[5], self.content_fake_B[5]) * lambda_C 
            self.loss_content_B = self.criterionIdt(self.content_B[5], self.content_fake_A[5]) * lambda_C 


        self.loss_style_B = self.criterionIdt(self.style_fake_B[5], self.style_B[5]) * lambda_C * 5

        

        self.loss_G += self.loss_style_B #+ self.loss_content_A + self.loss_content_B 
        self.loss_G.backward()

    def optimize_parameters(self, opt):
        # forward
        self.forward()
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)

        self.set_requires_grad([self.netG_E_A, self.netG_E_B, self.netG_D_A, self.netG_D_B], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        if self.opt.which_model_netD == 'fd':
            self.backward_FD_A()
            self.backward_FD_B()
        elif self.opt.which_model_netD == 'cd':
            self.backward_CD_A()
            self.backward_CD_B()
        else:
            self.backward_D_A()
            self.backward_D_B()
        self.optimizer_D.step()
