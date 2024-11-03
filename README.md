[合集 \- MindSponge教程(13\)](https://github.com)[1\.MindSponge分子动力学模拟——软件架构（2023\.08）2023\-08\-28](https://github.com/dechinphy/p/structure.html)[2\.MindSponge分子动力学模拟——安装与使用（2023\.08）2023\-08\-17](https://github.com/dechinphy/p/ms-system.html)[3\.MindSponge分子动力学模拟——定义一个分子系统（2023\.08）2023\-08\-30](https://github.com/dechinphy/p/mol-system.html)[4\.MindSponge分子动力学模拟——计算单点能（2023\.08）2023\-08\-31](https://github.com/dechinphy/p/single-point-energy.html)[5\.MindSponge分子动力学模拟——使用迭代器进行系统演化（2023\.09）2023\-09\-04](https://github.com/dechinphy/p/updater-md.html)[6\.MindSponge分子动力学模拟——Constraint约束（2023\.09）2023\-09\-06](https://github.com/dechinphy/p/constraint.html)[7\.MindSponge分子动力学模拟——定义Collective Variables（2024\.02）02\-19](https://github.com/dechinphy/p/cv.html)[8\.MindSponge分子动力学模拟——使用MDAnalysis工具进行后分析（2024\.02）02\-29](https://github.com/dechinphy/p/18042865/mda-mds)[9\.MindSponge分子动力学模拟——自建力场（2024\.03）03\-22](https://github.com/dechinphy/p/18089928/energy-cell):[飞数机场](https://ze16.com)[10\.MindSponge分子动力学模拟——自定义控制器（2024\.05）05\-15](https://github.com/dechinphy/p/18096072/controller)[11\.MindSponge分子动力学模拟——体系控制（2024\.05）05\-24](https://github.com/dechinphy/p/18210122/mscontrol)[12\.MindSponge分子动力学模拟——多路径分子模拟（2024\.05）05\-27](https://github.com/dechinphy/p/18216025/multi-md)13\.MindSponge分子动力学模拟——增强采样（2024\.11）11\-01收起
# 技术背景


关于增强采样（Enhanced Sampling）算法的具体原理，这里暂不做具体介绍，感兴趣的童鞋可以直接参考下这篇综述文章：[Enhanced sampling in molecular dynamics](https://github.com)。大致的作用就是，通过统计力学的方法，使得目标分子的CV（Collective Variables）具有一个尽可能大的采样子空间，并且可以将其还原回真实的自由能面。常用的增强采样算法，有早期的伞形采样，到后来大家常用的MetaDynamics以及高老师的温度积分增强采样算法（ITS）。在MindSponge中已经实现了MetaDynamics算法和ITS算法，本文我们使用MetaDynamics算法来做一个演示。


# 准备工作


使用MindSponge，可以参考[本系列文章](https://github.com)先了解一下MindSponge的安装和基本使用方法。这里我们用一个比较简单的多肽体系进行一个测试，相应的pdb文件为：



```
CRYST1    0.000    0.000    0.000  90.00  90.00  90.00 P 1           1
ATOM      1  H1  ACE A   1      -1.838  -6.570  -0.492  0.00  0.00      
     
ATOM      2  CH3 ACE A   1      -0.764  -6.587  -0.283  0.00  0.00      
     
ATOM      3  H2  ACE A   1      -0.392  -7.533  -0.746  0.00  0.00      
     
ATOM      4  H3  ACE A   1      -0.592  -6.446   0.740  0.00  0.00      
     
ATOM      5  C   ACE A   1      -0.006  -5.404  -0.828  0.00  0.00      
     
ATOM      6  O   ACE A   1      -0.544  -4.619  -1.673  0.00  0.00      
     
ATOM      7  N   ALA A   2       1.278  -5.323  -0.423  0.00  0.00      
     
ATOM      8  H   ALA A   2       1.622  -5.845   0.368  0.00  0.00      
     
ATOM      9  CA  ALA A   2       2.284  -4.164  -0.399  0.00  0.00      
     
ATOM     10  HA  ALA A   2       2.098  -3.653   0.505  0.00  0.00      
     
ATOM     11  CB  ALA A   2       3.651  -4.787  -0.566  0.00  0.00      
     
ATOM     12  HB1 ALA A   2       4.274  -4.031  -0.972  0.00  0.00      
     
ATOM     13  HB2 ALA A   2       3.977  -5.106   0.419  0.00  0.00      
     
ATOM     14  HB3 ALA A   2       3.697  -5.612  -1.274  0.00  0.00      
     
ATOM     15  C   ALA A   2       1.995  -3.152  -1.576  0.00  0.00      
     
ATOM     16  O   ALA A   2       1.544  -2.065  -1.221  0.00  0.00      
     
ATOM     17  N   NME A   3       2.255  -3.614  -2.845  0.00  0.00      
     
ATOM     18  H   NME A   3       2.788  -4.485  -2.929  0.00  0.00      
     
ATOM     19  CH3 NME A   3       1.991  -2.802  -4.055  0.00  0.00      
     
ATOM     20 HH31 NME A   3       2.561  -1.891  -3.988  0.00  0.00      
     
ATOM     21 HH32 NME A   3       1.897  -3.419  -4.937  0.00  0.00      
     
ATOM     22 HH33 NME A   3       0.985  -2.388  -3.930  0.00  0.00      

END

```

该构象可以保存成pdb文件然后直接用VMD进行可视化：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241101145533690-1703999747.png)

其他的力场文件我们直接使用MindSponge已经自带的`amber.ff14sb`力场即可。


# 普通MD案例


在测试增强采样算法之前，我们可以先跑一段普通的分子模拟测试一下：



```
from mindspore import nn, context
import mindspore as ms
# 固定随机种子，确保朗之万控温的随机数可以被复现
ms.set_seed(0)
# 这里使用的是一个未编译版本的mindsponge，所以要把sponge所在路径添加到系统路径中
import sys
sys.path.insert(0, '../..')
from sponge import ForceField, Sponge, set_global_units, Protein, UpdaterMD, WithEnergyCell
from sponge.callback import RunInfo, WriteH5MD
from sponge.colvar import Torsion
from sponge.function import PI

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
set_global_units('A', 'kcal/mol')

system = Protein('mol.pdb', template=['protein0.yaml'], rebuild_hydrogen=False)
energy = ForceField(system, parameters=['AMBER.FF14SB'], use_pme=False)
min_opt = nn.Adam(system.trainable_params(), 1e-03)

dihedrals = Torsion([[4, 6, 8, 14], [6, 8, 14, 16]])
run_info = RunInfo(20)

opt = UpdaterMD(system=system,
                time_step=1e-3,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin')

sim = WithEnergyCell(system, energy)
md = Sponge(sim, optimizer=opt, metrics={'dihedrals': dihedrals})
cb_h5md = WriteH5MD(system, 'test_meta.h5md', save_freq=10, write_image=False)
md.run(5000, callbacks=[run_info, cb_h5md])

```

运行之后会在本地保存一个h5md格式的轨迹文件，可以在vscode中使用h5web拓展工具打开：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241101145854757-1483006477.png)

找到dihedral的value，用matrix的形式查看，然后可以导出csv格式（如：`path_sink.csv`）：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241101150121876-54222554.png)

导出csv之后，可以使用如下的python脚本进行绘图：



```
import numpy as np
import matplotlib.pyplot as plt

def gaussian2(x1, x2, sigma1=1.0, sigma2=1.0, A=0.5):
    return np.sum(A*np.exp(-0.5*(x1**2/sigma1**2+x2**2/sigma2**2))/np.pi/sigma1/sigma2, axis=-1)

def potential_energy(position, psi, phi, sigma1, sigma2):
    # (A, )
    psi_, phi_ = position[:, 0], position[:, 1]
    # (A, R)
    delta_psi = psi_[:, None] - psi[None]
    delta_phi = phi_[:, None] - phi[None]
    # (A, )
    Z = -np.log(gaussian2(delta_psi, delta_phi, sigma1=sigma1, sigma2=sigma2, A=2.0)+1)
    return Z

data = np.genfromtxt('./path_sink.csv', delimiter=',')
phi = data[:, 0]
psi = data[:, 1]

num_grids = 100
num_levels = 10
psi_grids = np.linspace(-np.pi, np.pi, num_grids)
phi_grids = np.linspace(-np.pi, np.pi, num_grids)
grids = np.array(np.meshgrid(psi_grids, phi_grids)).T.reshape((-1, 2))

Z = potential_energy(grids, phi, psi, 1.0, 1.0).reshape((psi_grids.shape[0], phi_grids.shape[0])).T
X,Y = np.meshgrid(psi_grids, phi_grids)
levels = np.linspace(np.min(Z), np.max(Z), num_levels)

plt.figure()
plt.title("Biased MD Traj")
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\psi$')
fc = plt.contourf(X, Y, Z, cmap='Greens', levels=levels)
plt.colorbar(fc)

plt.xlim(-np.pi, np.pi)
plt.ylim(-np.pi, np.pi)
plt.plot(phi, psi, 'o', alpha=0.4, color='red')
plt.savefig('meta.png')

```

画出来的轨迹效果如下图所示：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241101093228541-1715853793.png)

可以发现，在不加任何的Bias的时候，整个轨迹还是比较集中的在一个区域，一般该区域就是对应了一个能量极小值点附近的区域。


# MetaDynamics案例


从上一个章节的结果中可以看出，常规的分子模拟采样方法很容易陷入到一个局部区域中，这样使得我们很难可以观测到其他采样子空间所对应的构象和相关信息，因此我们可以引入一个MetaDynamics增强采样算法，做一个有偏估计：



```
from mindspore import nn, context
import mindspore as ms
ms.set_seed(0)
import sys
sys.path.insert(0, '../..')
from sponge import ForceField, Sponge, set_global_units, Protein, UpdaterMD, WithEnergyCell
from sponge.callback import RunInfo, WriteH5MD
from sponge.colvar import Torsion
from sponge.function import PI
from sponge.sampling import Metadynamics

context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
set_global_units('A', 'kcal/mol')

system = Protein('mol.pdb', template=['protein0.yaml'], rebuild_hydrogen=False)
energy = ForceField(system, parameters=['AMBER.FF14SB'], use_pme=False)
min_opt = nn.Adam(system.trainable_params(), 1e-03)

# 这里定义的CV是phi和psi角，是一个2维的CV
dihedrals = Torsion([[4, 6, 8, 14], [6, 8, 14, 16]])
run_info = RunInfo(20)

# 配置Meta的参数，主要是高斯波包的高度、宽度、更新频率、CV范围、CV格点数等等
metad = Metadynamics(
    colvar=dihedrals,
    update_pace=10,
    height=2.5,
    sigma=0.4,
    grid_min=-PI,
    grid_max=PI,
    grid_bin=50,
    temperature=300,
    bias_factor=100,
    use_cutoff=True,
)

opt = UpdaterMD(system=system,
                time_step=1e-3,
                integrator='velocity_verlet',
                temperature=300,
                thermostat='langevin')

sim = WithEnergyCell(system, energy, bias=metad)
md = Sponge(sim, optimizer=opt, metrics={'dihedrals': dihedrals})
cb_h5md = WriteH5MD(system, 'test_meta.h5md', save_freq=10, write_image=False)
md.run(5000, callbacks=[run_info, cb_h5md])

```

用类似的方法，可以计算得增强采样之后的CV轨迹：



![](https://img2024.cnblogs.com/blog/2277440/202411/2277440-20241101092040118-1754988128.png)

可以看到，采样子空间在Meta的作用下已经扩展到几乎整个CV空间，这使得我们可以更快的去分析整个采样空间各处的自由能的相对大小。


# 总结概要


本文介绍了在MindSponge中进行分子动力学模拟以及增强采样的实现方法。通过使用MetaDynamics增强采样算法，我们可以将分子模拟的采样子空间，从某个能量极小值区域，扩大到尽可能大的采样子空间。


# 版权声明


本文首发链接为：[https://github.com/dechinphy/p/mindsponge\-meta.html](https://github.com)


作者ID：DechinPhy


更多原著文章：[https://github.com/dechinphy/](https://github.com)


请博主喝咖啡：[https://github.com/dechinphy/gallery/image/379634\.html](https://github.com)


