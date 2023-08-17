import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    data1 = np.load("train_nu.npz")
    nu = data1['nu_1d']
    print('nu is', nu)

    data = np.load('pred_poiseuille_para.npz')
    mesh = data['mesh']
    u = data['u']
    ut = data['ut']
    uMaxP = data['uMaxP']
    uMaxA = data['uMaxA']

    Ny, Nx, Np = u.shape
    idx_X = int(round(Nx / 2))
    y = np.linspace(-0.05, 0.05, 50)

    return nu, mesh, u, ut, uMaxP, uMaxA, idx_X, y

def plot_uProfiles(nu, u, ut, idx_X, y):
    can = [3, 6, 14, 49]
    ytext = [0.45, 0.28, 0.1, 0.01]
    plt.figure()
    plt.clf()
    ax1 = plt.subplot(111)
    Re = []

    for idx, idxP in enumerate(can):
        pT, = plt.plot(y, ut[:, idx_X, idxP], color='darkblue', linestyle='-', lw=3.0, alpha=1.0)
        pP, = plt.plot(y, u[:, idx_X, idxP], color='red', linestyle='--', dashes=(5, 5), lw=2.0, alpha=1.0)
        tmpRe = np.max(u[:, idx_X, idxP]) * 0.1 / nu[idxP]
        Re.append(tmpRe)
        nu_current = float("{0:.5f}".format(nu[idxP]))
        plt.text(-0.012, ytext[idx], r'$\nu = $' + str(nu_current), {'color': 'k', 'fontsize': 16})

    ax1.set_xlabel(r'$y$', fontsize=16)
    ax1.set_ylabel(r'$u(y)$', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    ax1.set_xlim([-0.05, 0.05])
    ax1.set_ylim([0.0, 0.62])
    plt.savefig('pipe_uProfiles_nuIdx_.png', bbox_inches='tight')
    
    return Re

def plot_pdf(uMaxA, uMaxP):
    plt.figure()
    ax1 = plt.subplot(111)
    sns.kdeplot(uMaxA[0, :], fill=True, label='Analytical', linestyle="-", linewidth=3)
    sns.kdeplot(uMaxP[0, :], fill=False, label='DNN', linestyle="--", linewidth=3.5, color='darkred')
    plt.legend(prop={'size': 16})
    ax1.set_xlabel(r'$u_c$', fontsize=16)
    ax1.set_ylabel(r'PDF', fontsize=16)
    ax1.tick_params(axis='both', labelsize=16)
    plt.savefig('pipe_unformUQ.png', bbox_inches='tight')
    plt.show()

def main():
    nu, mesh, u, ut, uMaxP, uMaxA, idx_X, y = load_data()
    Re = plot_uProfiles(nu, u, ut, idx_X, y)
    print('Re is', Re)
    np.savez('test_Re', Re=Re)
    plot_pdf(uMaxA, uMaxP)

if __name__ == "__main__":
    main()