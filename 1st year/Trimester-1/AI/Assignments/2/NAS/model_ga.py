import torch, random, os, json
import torch.nn as nn
from torch.optim import AdamW
from copy import deepcopy
import numpy as np
from model_cnn import CNN
import torch
import torch.nn as nn
from torch.optim import AdamW
# Define the search space for CNN architecture
class CNNSearchSpace:
    def __init__(self):
        self.conv_layers = [1, 2, 3, 4]
        self.filters = [16, 32, 64, 128]
        self.kernel_sizes = [3, 5, 7]
        self.pool_types = ['max', 'avg']
        self.activations = ['relu', 'leaky_relu']
        self.fc_units = [64, 128, 256, 512]

# Encode architecture as a chromosome (gene representation)
class Architecture:
    def __init__(self, genes=None):
        if genes is None:
            self.genes = self.random_genes()
        else:
            self.genes = genes
        self.fitness = 0
        self.accuracy = 0
        self.best_epoch = 0
    
    def random_genes(self):
        space = CNNSearchSpace()
        num_conv = random.choice(space.conv_layers)
        
        genes = {
            'num_conv': num_conv,
            'conv_configs': [],
            'pool_type': random.choice(space.pool_types),
            'activation': random.choice(space.activations),
            'fc_units': random.choice(space.fc_units)
        }
        
        for _ in range(num_conv):
            genes['conv_configs'].append({
                'filters': random.choice(space.filters),
                'kernel_size': random.choice(space.kernel_sizes)
            })
        
        return genes
    
    def __repr__(self):
        return f"Arch(conv={self.genes['num_conv']}, acc={self.accuracy:.4f})"

# Genetic Algorithm Operations
class GeneticAlgorithm:
    def __init__(self, population_size=20, generations=10, mutation_rate=0.2, crossover_rate=0.7):
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_architecture = None
        self.search_space = CNNSearchSpace()
    
    def initialize_population(self):
        self.population = [Architecture() for _ in range(self.population_size)]
    def evaluate_fitness(self, architecture, train_loader, val_loader, device, epochs=100,
                     w_conv=0.7, w_fc=0.3, lambda_penalty=0.05, param_scale=1e6):



        try:
            model = CNN(architecture.genes).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=0.001)

            # Quick training with early stopping (same logic as original)
            best_acc = 0.0
            patience = 10
            step = 1
            best_epoch = 1
            for epoch in range(1, epochs + 1):
                model.train()
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                # Validation
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += labels.size(0)

                val_acc = (correct / total) if total > 0 else 0.0

                if val_acc > best_acc:
                    step = 0
                    best_acc = val_acc
                    best_epoch = epoch
                else:
                    step += 1
                if step >= patience:
                    break

            # ---------- NEW: count params by layer type ----------
            conv_params = 0
            fc_params = 0
            other_params = 0
            for module in model.modules():
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    for p in module.parameters(recurse=False):
                        conv_params += p.numel()
                elif isinstance(module, (nn.Linear,)):
                    for p in module.parameters(recurse=False):
                        fc_params += p.numel()
                else:
                    for p in module.parameters(recurse=False):
                        other_params += p.numel()
            # ------------------------------------------------------

            # normalize counts (in millions)
            conv_M = conv_params / param_scale
            fc_M = fc_params / param_scale

            # weighted penalty
            penalty = (w_conv * conv_M) + (w_fc * fc_M)

            # final fitness = accuracy - lambda * penalty
            fitness = best_acc - (lambda_penalty * penalty)

            # update architecture metadata (keeps old behavior)
            architecture.accuracy = best_acc
            architecture.best_epoch = best_epoch
            architecture.fitness = fitness

            # cleanup
            del model, inputs, outputs, labels
            torch.cuda.empty_cache()

            # For logging (ensure logs show conv/fc counts & penalty)
            print(f"[FITNESS] acc={best_acc:.4f} conv_params={conv_params} fc_params={fc_params} "
                f"penalty={penalty:.6f} lambda={lambda_penalty:.6f} fitness={fitness:.6f}", flush=True)

            return fitness

        except Exception as e:
            print(f"Error evaluating architecture: {e}", flush=True)
            architecture.fitness = 0.0
            architecture.accuracy = 0.0
            return 0.0

    
    def roulette_wheel_selection(self, k=None):
        population = self.population
        pop_size = len(population)
        if pop_size == 0:
            return []
        if k is None:
            k = self.population_size

        # collect fitness values (ensure float)
        fitnesses = np.array([float(getattr(ind, 'fitness', 0.0)) for ind in population], dtype=float)

        # handle non-finite values
        if not np.all(np.isfinite(fitnesses)):
            fitnesses = np.nan_to_num(fitnesses, nan=0.0, posinf=0.0, neginf=0.0)

        # if any fitness <= 0, shift to make positive (preserve proportionality)
        min_f = fitnesses.min()
        if min_f <= 0:
            fitnesses = fitnesses - min_f + 1e-8

        total = fitnesses.sum()
        if total <= 0:
            probs = np.ones(pop_size) / pop_size
        else:
            probs = fitnesses / total

        # sample indices with the probability vector
        indices = np.random.choice(pop_size, size=k, replace=True, p=probs)

        return [deepcopy(population[i]) for i in indices]

    
    def crossover(self, parent1, parent2):
        """Single-point crossover for architectures"""
        if random.random() > self.crossover_rate:
            return deepcopy(parent1), deepcopy(parent2)
        
        child1_genes = deepcopy(parent1.genes)
        child2_genes = deepcopy(parent2.genes)
        
        # Crossover number of conv layers and pool type
        if random.random() < 0.5:
            child1_genes['num_conv'], child2_genes['num_conv'] = child2_genes['num_conv'], child1_genes['num_conv']
        
        # Crossover pool type and activation
        if random.random() < 0.5:
            child1_genes['pool_type'], child2_genes['pool_type'] = child2_genes['pool_type'], child1_genes['pool_type']
            child1_genes['activation'], child2_genes['activation'] = child2_genes['activation'], child1_genes['activation']
        
        # Adjust conv_configs to match num_conv
        min_len = min(child1_genes['num_conv'], len(child1_genes['conv_configs']))
        child1_genes['conv_configs'] = child1_genes['conv_configs'][:min_len]
        while len(child1_genes['conv_configs']) < child1_genes['num_conv']:
            child1_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        min_len = min(child2_genes['num_conv'], len(child2_genes['conv_configs']))
        child2_genes['conv_configs'] = child2_genes['conv_configs'][:min_len]
        while len(child2_genes['conv_configs']) < child2_genes['num_conv']:
            child2_genes['conv_configs'].append({
                'filters': random.choice(self.search_space.filters),
                'kernel_size': random.choice(self.search_space.kernel_sizes)
            })
        
        return Architecture(child1_genes), Architecture(child2_genes)
    
    def mutation(self, architecture):
        """Mutate architecture genes"""
        if random.random() > self.mutation_rate:
            return architecture
        
        genes = deepcopy(architecture.genes)
        mutation_type = random.choice(['conv_param', 'num_layers', 'pool_activation', 'fc_units'])
        
        if mutation_type == 'conv_param' and genes['conv_configs']:
            # Mutate a random conv layer
            idx = random.randint(0, len(genes['conv_configs']) - 1)
            genes['conv_configs'][idx]['filters'] = random.choice(self.search_space.filters)
            genes['conv_configs'][idx]['kernel_size'] = random.choice(self.search_space.kernel_sizes)
        
        elif mutation_type == 'num_layers':
            # Change number of conv layers
            genes['num_conv'] = random.choice(self.search_space.conv_layers)
            # Adjust conv_configs
            if genes['num_conv'] > len(genes['conv_configs']):
                for _ in range(genes['num_conv'] - len(genes['conv_configs'])):
                    genes['conv_configs'].append({
                        'filters': random.choice(self.search_space.filters),
                        'kernel_size': random.choice(self.search_space.kernel_sizes)
                    })
            else:
                genes['conv_configs'] = genes['conv_configs'][:genes['num_conv']]
        
        elif mutation_type == 'pool_activation':
            genes['pool_type'] = random.choice(self.search_space.pool_types)
            genes['activation'] = random.choice(self.search_space.activations)
        
        elif mutation_type == 'fc_units':
            genes['fc_units'] = random.choice(self.search_space.fc_units)
        
        return Architecture(genes)
    
    def evolve(self, train_loader, val_loader, device, run=1):
        parent = os.path.abspath('')
        """Main evolutionary loop"""
        self.initialize_population()
        print(f"Starting with {self.population_size} Population:\n{self.population}\n", flush=True)
        
        for generation in range(self.generations):
            print(f"\n{'='*60}", flush=True)
            print(f"Generation {generation + 1}/{self.generations}", flush=True)
            print(f"{'='*60}", flush=True)
            
            # Evaluate fitness
            for i, arch in enumerate(self.population):
                print(f"Evaluating architecture {i+1}/{self.population_size}...", end=' ', flush=True)
                fitness = self.evaluate_fitness(arch, train_loader, val_loader, device)
                print(f"Fitness: {fitness:.4f}, Accuracy: {arch.accuracy:.4f}", flush=True)
            
            # Sort by fitness score
            print(f"\nSorting population in terms of fitness score (high -> low) ...", flush=True)
            self.population.sort(key=lambda x: x.fitness, reverse=True)
            
            # Track best
            if self.best_architecture is None or self.population[0].fitness > self.best_architecture.fitness:
                self.best_architecture = deepcopy(self.population[0])
            
            print(f"Best in generation: {self.population[0]}\n", flush=True)
            print(f"Best overall: {self.best_architecture}", flush=True)
            
            # Selection
            print(f"\nPerforming roulette_wheel_selection of total population: {self.population_size} ...", flush=True)
            selected = self.roulette_wheel_selection()
            
            # Crossover and Mutation
            print(f"Performing Crossover & Mutation ...", flush=True)
            next_generation = []
            
            # Elitism: keep top 2 architectures
            print(f"Elitism: Keeping top 2 architectures in next generation.", flush=True)
            next_generation.extend([deepcopy(self.population[0]), deepcopy(self.population[1])])
            
            while len(next_generation) < self.population_size:
                parent1 = random.choice(selected)
                parent2 = random.choice(selected)
                
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                
                next_generation.append(child1)
                if len(next_generation) < self.population_size:
                    next_generation.append(child2)
            
            self.population = next_generation
            print(f"Next Generation: {self.population}", flush=True)
            with open(os.path.join(parent, 'outputs', f'run_{run}', f"generation_{generation}.jsonl"), 'w') as f:
                for obj in self.population:
                    f.write(json.dumps(obj.genes))
        
        return self.best_architecture