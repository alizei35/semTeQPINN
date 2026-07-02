import pennylane as qp

def HardwareEfficient(angles,wires,n_wires,n_layers):
    for i in range(n_layers):
        for j in range(n_wires):
            qp.RX(angles[i,j,0],wires=wires[j])
            qp.RY(angles[i,j,1],wires=wires[j])
            qp.RZ(angles[i,j,2],wires=wires[j])
        for j in range(n_wires-1):
            qp.CNOT(wires=wires.subset([j,j+1]))

def HardwareEfficient_cat(angles,wires,n_wires,n_layers):
    from catalyst import for_loop
    @for_loop(0,n_layers,1)
    def loop_layers(i):
        @for_loop(0,n_wires,1)
        def loop_qubits_rot(j):
            qp.RX(angles[i,j,0],wires=wires[j])
            qp.RY(angles[i,j,1],wires=wires[j])
            qp.RZ(angles[i,j,2],wires=wires[j])
        loop_qubits_rot()

        @for_loop(0,n_wires-1,1)
        def loop_qubits_cnot(j):
            qp.CNOT(wires=wires.subset([j,j+1]))
        loop_qubits_cnot()
    loop_layers()
