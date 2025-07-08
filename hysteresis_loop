import matplotlib.pyplot as plt
import numpy as np
import os  

def Gsec_Damping(shear_strain, z_xz):
    # (Assuming the internal details of this function remain as in your original code)
    # Find indices of max and min shear strain values
    max_index = np.argmax(shear_strain)
    min_index = np.argmin(shear_strain)
    
    # Compute Gsec
    delta_strain = shear_strain[max_index] - shear_strain[min_index]
    delta_stress = z_xz[max_index] - z_xz[min_index]
    gsec = delta_stress / delta_strain
    
    # Compute damping
    delta_w = np.trapz(z_xz, shear_strain)
    maxindex = np.argmax(shear_strain)
    maxstrain = shear_strain[maxindex]
    maxstress = z_xz[maxindex]
    W = 0.5 * maxstrain * maxstress
    Damping = delta_w / (4 * np.pi * W)
    return gsec, Damping

with open('cyclic_stress_NO1.txt', 'r') as f:
    lines = f.readlines()
    
lines = lines[2:]

Gmax_stress = []
Gmax_shear_strain = []

for line in lines:
    # Splitting by comma to get individual columns
    columns = line.strip().split(',')
    Gmax_stress.append(float(columns[11]))  # third column is z_force
    Gmax_shear_strain.append(float(columns[6]))  # fifth column is shear_strain
Gmax=abs(Gmax_stress[0])/abs(Gmax_shear_strain[0])




# Main loop starts here
for i in range(1, 22):
    filename = f'cyclic_stress_NO{i}.txt'
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"File not found: {filename}")
        continue
    
    lines = lines[2:]

    Gmax_stress = []
    Gmax_shear_strain = []

    for line in lines:
        # Splitting by comma to get individual columns
        columns = line.strip().split(',')
        Gmax_stress.append(float(columns[11]))  # third column is z_force
        Gmax_shear_strain.append(float(columns[6]))  # fifth column is shear_strain
    Gmax=abs(Gmax_stress[0])/abs(Gmax_shear_strain[0])
    # Removing the first two lines which are comments
    print(f'cyclic_stress_NO{i}-Gmax : {Gmax}')

    for start_step in range(51000000, 131000000, 20000000):
        z_xz = []
        shear_strain = []
        steps_in_range = []

        for line in lines:
            columns = line.strip().split(',')
            step = int(columns[0])

            if start_step <= step < start_step + 20000000:
                z_xz.append(float(columns[11]))
                shear_strain.append(float(columns[6]))
                steps_in_range.append(step)

        if not z_xz or not shear_strain:
            print(f"No data in the specified step range ({start_step}-{start_step+20000000}) for file: {filename}")
            continue

        gsec, D = Gsec_Damping(shear_strain, z_xz)
        print(f"cyclic_stress_NO{i} (Steps: {start_step}-{start_step+20000000}): Gsec: {gsec}, Damping: {D}")

        # Plotting the data
        plt.plot(shear_strain, z_xz, label=f"cyclic-NO{i} ")
        plt.ylabel('Z-XZ')
        plt.xlabel('Shear Strain')
        plt.title(f'cyclic_stress_NO{i} (Steps: {start_step}-{start_step+20000000})')
        plt.legend()
        plt.grid(True)
        save_directory = 'graph_images'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)  # 지정된 디렉토리가 없으면 새로 만듭니다.
        
        save_filename = f"{save_directory}/cyclic_stress_NO{i}_Steps_{start_step}_to_{start_step+20000000}.png"
        
        # 그래프를 이미지 파일로 저장합니다.
        plt.savefig(save_filename)
        print(f"Graph saved as {save_filename}")  # 파일이 저장된 것을 사용자에게 알립니다.

        # 그래프를 화면에 표시하지 않고 종료합니다.
        plt.close()
        #plt.show()



#히스테리시스루프 개별적으로 그리는 코드
for i in range(1, 22):
    filename = f'cyclic_stress_NO{i}.txt'
    # Assuming the data is saved as 'data.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Removing the first two lines which are comments
    lines = lines[2:]

    z_xz = []
    shear_strain = []

    for line in lines:
        # Splitting by comma to get individual columns
        columns = line.strip().split(',')
        z_xz.append(float(columns[11]))  # third column is z_force
        shear_strain.append(float(columns[6]))  # fifth column is shear_strain

    def compute_gsec_and_damping(shear_strain, z_xz):
        # Find indices of max and min shear strain values
        max_index = np.argmax(shear_strain)
        min_index = np.argmin(shear_strain)
        
        # Compute Gsec
        delta_strain = shear_strain[max_index] - shear_strain[min_index]
        delta_stress = z_xz[max_index] - z_xz[min_index]
        gsec = delta_stress / delta_strain
        
        # Compute damping
        loop_area = np.trapz(z_xz, shear_strain)  # Area under the curve using trapezoidal rule
        max_stress = np.max(z_xz)
        max_strain = np.max(shear_strain)
        damping = loop_area / (max_stress * max_strain/2)

        return gsec, damping

    # Example using your shear_strain and z_xz data
    gsec, damping = compute_gsec_and_damping(shear_strain, z_xz)
    print(f"Gsec: {gsec}, Damping: {damping}")   

    # Plotting the data
    plt.plot(shear_strain, z_xz, label='Shear Strain vs. Z-XZ')
    plt.ylabel('Z-XZ')
    plt.xlabel('Shear Strain')
    plt.title('Shear Strain vs. Z-Force Graph')
    plt.legend()
    plt.grid(True)
    plt.show()
