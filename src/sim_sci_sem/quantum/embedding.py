import pennylane as qp

def Trainable(angles,x,wires):
    qp.AngleEmbedding(features=angles*x,wires=wires,rotation='Y')

def Trainable_cat(angles,x,wires,n_wires):
    from catalyst import for_loop
    @for_loop(0,n_wires,1)
    def loop_wires(i):
        qp.RY(angles[i]*x,wires=wires[i])
    loop_wires()
