# TSP Project
This project builds a small TSP model. Big O (1) modifications and edits are made on TSP solutions generated. Different optimization methods are tested ranging from models with 10 cities to 200 cities. 

When creating initial solutions, various methods were used. 
- Creating solutions at random
- Creating solution by choosing the closest city (Nearest Neighbour/ Greedy)
- Creating solutions by connecting the shortest edges first (Shortest Edge/ Greedy)

The following methods were used to optimize solutions, 
- the First Improvement Search which uses the first improvement found in the current solution 
- Best Improvement Search which looks for the current best improvement to the solution 
Each Search used a swap and insert approach

However, the search may be stuck in local optimas so certain metaheuristics are used to escape. 
- Iterated Local Search (ILS) with Double-Bridge perturbation which "cuts" the solution into piece and reconnects in different order
- Threshold Search which will accept worse moves of a certain threshold to escape local optimas and find new neighbourhoods which may have the optimal solution

Using these methods, different local optimas are searched and eventually an optimal solution will be found.

Conclusion

For small to medium input sizes methods do not matter
However for larger input sizes, threshold algorithm is significantly superior compared to ILS due to ILS having to search many local optimas unlike threshold search 
