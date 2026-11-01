"""Fixed-shape full training replay; computation and optimizer stay unchanged.

Construct/warm the model and capture on the same explicit CUDA stream. The
caller supplies an already initialized capturable optimizer and static input.
No shape-dependent kernel selection or change to saved activation policy.
"""
import torch


class GraphedTrainingStep:
    def __init__(self,model,optimizer,loss_fn,implementation):
        self.model=model
        self.optimizer=optimizer
        self.parameters=tuple(model.parameters())
        self.stream=torch.cuda.current_stream()
        self.graph=torch.cuda.CUDAGraph()
        self.events=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(4)]
        optimizer.zero_grad(set_to_none=True)
        with implementation(),torch.cuda.graph(self.graph,stream=self.stream):
            self.events[0].record()
            self.loss=loss_fn()
            self.events[1].record()
            self.loss.backward()
            self.events[2].record()
            optimizer.step()
            self.events[3].record()
        self.gradients=tuple(p.grad for p in self.parameters)

    def replay(self):
        # Eager control runs may have replaced .grad; expose the actual graph
        # outputs again. All captured gradient writes overwrite this step's data.
        for p,g in zip(self.parameters,self.gradients):
            if p.grad is not g:p.grad=g
        self.graph.replay()
        return self.loss

    def phase_ms(self):
        return {name:self.events[i].elapsed_time(self.events[i+1])
                for i,name in enumerate(('forward_loss','backward','optimizer'))}
