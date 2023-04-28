import os, glob
import random
import torch
from PIL import Image
from natsort import natsorted

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class ImageFolderWithPaths():

    def __init__(self, root, transform=None):

        self.transform = transform

        files = []
        for ext in IMG_EXTENSIONS:
            for f in glob.glob(f'{root}/*{ext}'):
                files += [os.path.split(f)[1]]

        self.path = []
        self.fnames = []
        for f in natsorted(files):
            self.path.append(f'{root}/{f}')
            self.fnames.append(f)

        self.datanum = len(files)
        
    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        sample = Image.open(self.path[index]).convert('RGB')
        fname = self.fnames[index]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, fname


class ImageFolder2ClassesWithPathsLoader():

    def __init__(self, root1, root2, batch_size, transform=None):

        self.transform = transform
        self.batch_size = batch_size
        
        # Class 1
        files1 = []
        for ext in IMG_EXTENSIONS:
            for f in glob.glob(f'{root1}/*{ext}'):
                files1 += [os.path.split(f)[1]]

        self.path1 = []
        self.fnames1 = []
        for f in natsorted(files1):
            self.path1.append(f'{root1}/{f}')
            self.fnames1.append(f)

        self.datanum1 = len(files1)
        
        # Class 2
        files2 = []
        for ext in IMG_EXTENSIONS:
            for f in glob.glob(f'{root2}/*{ext}'):
                files2 += [os.path.split(f)[1]]

        self.path2 = []
        self.fnames2 = []
        for f in natsorted(files2):
            self.path2.append(f'{root2}/{f}')
            self.fnames2.append(f)

        self.datanum2 = len(files2)
        
    def __call__(self):
    
        out1 = []
        out2 = []
        out3 = []
        out4 = []
    
    	# tensor1
        data1 = random.choices(range(self.datanum1), k=self.batch_size)
        for idx in data1:
            image = Image.open(self.path1[idx]).convert('RGB')
            out1 += [self.transform(image)]
            out3 += [self.fnames1[idx]]
        out1 = torch.stack(out1, dim=0)
        
    	# tensor2
        data2 = random.choices(range(self.datanum2), k=self.batch_size)
        for idx in data2:
            image = Image.open(self.path2[idx]).convert('RGB')
            out2 += [self.transform(image)]
            out4 += [self.fnames2[idx]]
        out2 = torch.stack(out2, dim=0)

        return out1, out2, out3, out4


class ImageFolderGMMWithPathsLoader():

    def __init__(self, rootN, rootGMM, batch_size, transform=None):

        self.transform = transform
        self.batch_size = batch_size        
        
        # Class 'Normal'
        filesN = []
        for ext in IMG_EXTENSIONS:
            for f in glob.glob(f'{rootN}/*{ext}'):
                filesN += [os.path.split(f)[1]]

        self.pathN = []
        self.fnamesN = []
        for f in natsorted(filesN):
            self.pathN.append(f'{rootN}/{f}')
            self.fnamesN.append(f)

        self.datanumN = len(filesN)
        
        # Class 'GMM'
        filesGMM = []
        for f in glob.glob(f'{rootGMM}/*'):
            filesGMM += [os.path.split(f)[1]]

        self.pathGMM = []
        self.fnamesGMM = []
        for f in natsorted(filesGMM):
            self.pathGMM.append(f'{rootGMM}/{f}')
            self.fnamesGMM.append(f)

        self.datanumGMM = len(filesGMM)
        
    def __call__(self, device):
    
        out1 = []
        out2 = ([], [], [], [], [], [], [], [])
        out3 = []
        out4 = []
    
    	# tensor1
        data1 = random.choices(range(self.datanumN), k=self.batch_size)
        for idx in data1:
            image = Image.open(self.pathN[idx]).convert('RGB')
            out1 += [self.transform(image)]
            out3 += [self.fnamesN[idx]]
        out1 = torch.stack(out1, dim=0).to(device)
        
    	# tensor2
        data2 = random.choices(range(self.datanumGMM), k=self.batch_size)
        for idx in data2:
            gmm_params = torch.load(self.pathGMM[idx])
            addGMM(out2, gmm_params, device)
            out4 += [self.fnamesGMM[idx]]
        out2 = catGMM(out2, dim=0)

        return out1, out2, out3, out4


class ImageFolderGTWithPaths():

    def __init__(self, root1, root2, transform=None, transformGT=None):

        self.transform = transform
        self.transformGT = transformGT

        files = []
        for ext in IMG_EXTENSIONS:
            for f in glob.glob(f'{root1}/*{ext}'):
                files += [os.path.split(f)[1]]

        self.path1 = []
        self.path2 = []
        self.fnames = []
        for f in natsorted(files):
            self.path1.append(f'{root1}/{f}')
            self.path2.append(f'{root2}/{f}')
            self.fnames.append(f)

        self.datanum = len(files)
    
    def __len__(self):
        return self.datanum

    def __getitem__(self, index):
        sample1 = Image.open(self.path1[index]).convert('RGB')
        sample2 = Image.open(self.path2[index]).convert('RGB')
        fname = self.fnames[index]
        if self.transform is not None:
            sample1 = self.transform(sample1)
        if self.transformGT is not None:
            sample2 = self.transformGT(sample2)
        return sample1, sample2, fname


def getImageAndGMM(pathN, pathGMM, device, transform):
    imageN = Image.open(pathN).convert('RGB')
    imageN = transform(imageN).unsqueeze(dim=0).to(device)
    #########################################
    gmm_params = torch.load(pathGMM)
    color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
    gmm_params = (color.to(device), mu_x.to(device), mu_y.to(device), sigma_x.to(device), sigma_y.to(device), rho.to(device), pi.to(device), scale.to(device))
    #########################################
    return imageN, gmm_params


def addGMM(gmm_params, gmm_params_add, device):
    color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
    color_add, mu_x_add, mu_y_add, sigma_x_add, sigma_y_add, rho_add, pi_add, scale_add = gmm_params_add
    color += [color_add.to(device)]
    mu_x += [mu_x_add.to(device)]
    mu_y += [mu_y_add.to(device)]
    sigma_x += [sigma_x_add.to(device)]
    sigma_y += [sigma_y_add.to(device)]
    rho += [rho_add.to(device)]
    pi += [pi_add.to(device)]
    scale += [scale_add.to(device)]
    return


def catGMM(gmm_params, dim=0):
    color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
    color = torch.cat(color, dim=dim)
    mu_x = torch.cat(mu_x, dim=dim)
    mu_y = torch.cat(mu_y, dim=dim)
    sigma_x = torch.cat(sigma_x, dim=dim)
    sigma_y = torch.cat(sigma_y, dim=dim)
    rho = torch.cat(rho, dim=dim)
    pi = torch.cat(pi, dim=dim)
    scale = torch.cat(scale, dim=dim)
    return color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale