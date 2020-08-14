# Molecule in a Box: A Roadmap from E to G



If you've taken a class in chemistry, you've doubtless idolized thermochemical terms like $G$.

If you've taken a class in electronic structure theory, all you probably got was $E$.

To add to the pain, classes that emphasize $G$ will often nicely cover the thermochemical derivations but are unlikely to address how to interface with actual electronic structure calculations!

**Unfortunately, I think many students (as I did) walk away not really knowing how to interface between E and G**. In many labs I've been in, a script just gets passed around with the promise of *this handles all the thermochemistry, don't worry about it*. 

If you are just starting out like I was, I hope this series of posts helps you gain some perspective on how everything fits together.

# Part 1: A Molecule is a Box

Kind of. What do I mean? I mean that as a pedagogical tool for thermochemistry, **it is helpful to think of the volume the molecule exists in as an inherent property of the molecule itself**. Let's get more concrete. Our molecule will have:

* Volume $V$
* A N-body wavefunction given by $|\psi_{nuc,elec}>$,
* A Hamiltonian, given by (in the non-relativistic case):

$$
H = T_n + T_e + U_{en} + U_{nn} + U_{ee}
$$

where we have in order the kinetic energy of the nuclei, electrons, and mix-match potentials between all the nuclei and electrons respectively.

### A Molecule in its Natural Habitat

Below is I hope a helpful image for thinking about the "molecule as a box" concept:

![image.png](2020-08-13-thermochem_files/att_00000.png)

The environment has several variables which define various "costs" for our molecule (kind of like universal "rent"):

* The temperature, T (*negative* cost of entropy)
* The pressure, P (cost of volume)
* The energy of the Hamiltonian, $H$ (cost of movement and interactions)

It's usually the case that the temperature and pressure are positive and so all molecules are encouraged to be:

* Highly entropic
* With minimal volume
* At low energy

What a balancing act! When we think we have a good answer (an energy, entropy, and volume for a set of particles in the box), we package the result in what we will call the "molecular chemical potential" (MCP, $\mu_{mol}$):

$$
\mu_{mol}(U_{mol},V_{mol},S_{mol}) = U_{mol} + PV_{mol} - TS_{mol}
$$

where the internal energy U can be decomposed into the so-called "electronic energy" $E_{mol}$, the zero-point vibrational energy ($ZPVE_{mol}$), and Boltzmann thermal contributions ($U_{T,mol}$):

$$
U_{mol} = E_{mol} + ZPVE_{mol} + U_{T,mol}
$$

Finally, if you take away anything from this post, let it be these four points:

1) **The output of an electronic structure package (e.g. Gaussian) is generally only the first term in the internal energy, $E_{mol}$**.

2) **The free energy of a *system* (several molecules) is simply the sum of its molecular chemical potentials**:

$$
G_{sys} = \sum_i^{molecules} N_i \mu_i
$$

3) **The molecular chemical potentials of identical molecules in the system are always identical**

4) **The goal of any system is to minimize its free energy**



## Chemical Reactions as Reactions of Boxes  

Okay, so, if molecules are boxes, how do reactions happen? Easier than you might think.

We will define a chemical reaction as follows:

* A chemical reaction is a transformation of a system from one set of boxes to another that conserves the number of nuclei and electrons.

This is quite a general process! Hence, chemistry is a very general topic.

### Box Disposal

Note that our definition of a chemical reaction is perfectly fine with destroying *boxes*, just not the particles inside them. For example, consider the reaction of two empty boxes:

![image.png](2020-08-13-thermochem_files/att_00001.png)

How do we calculate the "change in free energy" of this reaction? It is simple: **to calculate the free energy of a reaction, calculate the chemical potentials on both sides and take the difference**. 

Here, the internal energies and entropies are all zero, so we only need to worry about the volume terms:

$$
\Delta \mu_{rxn} = \mu_{products} - \mu_{reactants} = PV - 2PV = -PV
$$

So this reaction, of course, goes essentially to completion. This also means that an empty box will destroy itself:

![image.png](2020-08-13-thermochem_files/att_00002.png)

Where evidently,

$$
\Delta \mu_{rxn} = \mu_{products} - \mu_{reactants} = 0 - PV = -PV
$$

So, *empty boxes are inherently unstable*; big whoop.

Let's make things more interesting.

# Part 2: Proton in a Box

If you ask an electronic structure package for the energy of a proton in a box, it will tell you that it's zero:

```python
from pyscf import gto,scf
mol = gto.Mole()
mol.atom = [["H",(0,0,0)]]
mol.charge = 1
mol.build()
mf = scf.RHF(mol)
mf.kernel()
```

    converged SCF energy = 0





    0.0



Hater. There's of course much more going on here behind the scenes. Let's look at our Hamiltonian:

$$
H = T_n + T_e + U_{en} + U_{nn} + U_{ee}
$$

Clearly the last four terms all go to zero. But the kinetic energy of the proton is very much non-zero! So, where is the remaining energy? To find out, we will need to introduce some approximations:

## The Born-Oppenheimer Approximation

The reason electronic structure packages get away with nonsense like "the energy of a proton is zero" is because of the Born-Oppenheimer approximation. Mathematically, it is essentially approximation that

$$
|\psi_{nuc,elec}> = |\psi_{nuc}> \otimes |\psi_{elec}>
$$

More physically, it can be stated as:

* **Nuclei essentially exist on an energy surface defined by the electronic energy and the nuclear repulsion at any given point in 3N-space** (where N is the number of nuclei in the molecule)

So electronic structure packages make it their goal to define this surface. This surface is sometimes called the "potential energy surface",but this is a misnomer in my opinion because the energy clearly involves the kinetic energy of the electrons.

The intuition here is roughly that the electrons are much faster than the nuclei, and so we can effectively treat the electrons separately for any position of the nuclear coordinates and vice-versa. Our Hamiltonian is then also naturally split as a tensor product:

$$
H = H_{nuc} \otimes I + I \otimes H_{elec}
$$

(because energy is an additive quantum number), and so the energy is given as:

$$
E = E_{nuc} + E_{elec}
$$

Is the $E_{elec}$ in this expression equivalent to the $E_{elec}$ from an electronic structure package? Almost. The one exception is the nuclear repulsion energy, as this term also responds instantaneously to nuclear movement. To be concise:

$$
E_{SCF} = E_{elec} + U_{nn}
$$

where $E_{SCF}$ is meant to invoke "output of your computer program" (of course whether you used an SCF method or not is irrelevant).

So, how do we recover the rest of the nuclear energy, that our electronic structure program has so terribly forgotten? We simply follow the Born-Oppenheimer approximation to a t:

* **To get the rest of the nuclear energy, the wavefunctions of the nuclear degrees of freedom are solved for on a potential energy surface defined by $E_{SCF}$**

However, it turns out that the splitting between the levels of these degrees of freedom is quite small (comparable to kT). Thus, we will need to introduce:


## The Statistical Approximation

At finite temperatures, we need to start treating our molecule as a statistical distribution of states. The key assumption we make is that at conserved energy the system samples all all accessible states $|\psi_{nuc,elec}>_j$ with equal probability.

It turns out that **at a set *temperature*, this assumption actually means that states become exponentially less likely according to their energy**. We can describe the probability of any given state of our molecule as

$$
P(|\psi_{nuc,elec}>_j) = \frac{e^{\frac{-E_j}{kT}}}{\sum_{j} e^{\frac{-E_j}{kT}}}
$$

where we define the "molecular partition function" as:

$$
q_{mol} = \sum_{j} e^{\frac{-E_j}{kT}}
$$

which serves the role of the normalization constant in our probability expression above. Because the energies of different degrees of freedom are allowed to be independent of one another in the cannonical ensemble, we may write the sum as a product of sums over different degrees of freedom; for example:

$$
q_{mol} = \sum_{i} e^{\frac{-E_nuc,i}{kT}} \sum_j e^{\frac{-E_elec,j}{kT}} = q_{nuc} q_{elec}
$$

where we have defined separate partition functions for each of the degrees of freedom of the molecule. We care a lot about the molecular partition function, because it turns out that **all the terms in the molecular chemical potential may be derived directly from the partition function through the following equations:**

$$
U = k T^2 \frac{\partial \ln q}{\partial T}
$$

$$
S = k \ln q + k T \frac{\partial \ln q}{\partial T}
$$

For example, our electronic partition function is given by:

$$
q_{elec} (T,\epsilon_k) = \sum_k e^\frac{-\epsilon_k}{k T}
$$

where $\epsilon_k$ are the calculated energies of states in the molecule. Of course, this requires infinite states (and thus an infinite basis!). To circumvent this issue, we make the (quite good) approximation that the energy gaps between states is many times bigger than $kT$ and thus approximate the partition function as:

$$
q_{elec} (T,\epsilon_0) = g_0 e^\frac{-E_{elec}}{k T}
$$

where $g_0$ is the degeneracy of the ground state. Straightforward application of the equations for U and S give

$$
U_{elec} = E_{elec}
$$

$$
S_{elec} = k \ln g_0
$$

so indeed, **a triplet ground state is more entropically stable than a singlet!**

Here, we also see the critical connection between the output of an electronic structure calculation and the thermochemical treatment of the molecule: *the electronic energy makes a contribution to the internal energy via the electronic partition function*.

Of course, for our lonely proton, both these terms our zero– we must instead look to:


## The Partition Functions of Nuclear Motion

To a quite good approximation, we will take the historically successful approach and split up the nuclear degrees of freedom as follows:

$$
|\psi_{nuc}> = |\psi_{trans}> \otimes |\psi_{rot}> \otimes |\psi_{vib}>
$$

This allows us to split up the partition function as:

$$
q_{nuc} = q_{trans} q_{rot} q_{vib}
$$

which accounts for all the nuclear degrees of freedom of the molecule. The derivation of these partition functions from the quantum (while approximate) treatment of these degrees of freedom is one of the great triumphs of quantum statistical mechanics! Here, we take a major shortcut and cite the results from chapter 10 of "Computational Chemistry: Theories and Models" by Chris Cramer:

$$
q_{trans}(V,T) = (\frac{2\pi M k_B T}{h^2})^{3/2} V
$$

$$
q_{rot}(T) = \frac{\sqrt{\pi I_A I_B I_C}}{\sigma}(\frac{8 \pi^2 k_B T}{h^2})^{3/2}
$$

$$
q_{vib}(T,\omega_i) = e^\frac{-\sum_i h \omega_i}{2 k T}  \prod_{\omega_i} (\frac{1}{1-e^\frac{-h\omega_i}{kT}})
$$

**The most troublesome of these is the vibrational partition function**. This is for a few reasons:

* The vibrational partition function requires vibrational frequencies $\omega_i$ that are usually procured from diagonalization of the mass-weighted Hessian; an often technically challenging endeavor

* The harmonic oscillator approximation in which $q_{vib}(T)$ was derived is not great in many cases, and facile higher-level treatment is an open research topic

Directly applying our equations for U and S to these partition functions (you can check!) results in the analytically tractable forms for the internal energies:

$$
U_{trans} = \frac{3}{2} k T
$$

$$
U_{rot} = \frac{3}{2} k T
$$

$$
U_{vib} = \sum_i \frac{1}{2} h \omega_i + \sum_i \frac{h \omega_i}{(e^\frac{h \omega_i}{k T} - 1)}
$$

and the entropies:

$$
S_{trans} = k \ln \left( \left( \frac{2 \pi M k T}{h^2} \right)^\frac{3}{2} V \right) + k\frac{3}{2}
$$

$$
S_{rot} = k  \ln \left( \frac{\sqrt{\pi I_A I_B I_C}}{\sigma} \left( \frac{8 \pi^2 k T}{h^2} \right)^\frac{3}{2} \right) + k\frac{3}{2}
$$

$$
S_{vib} = \sum_i \frac{h \omega_i}{T(e^\frac{h \omega_i}{k T} - 1)} - \ln(1-e^\frac{-h \omega_i}{k T})
$$

A key for these expressions:

* k is the Boltzmann constant
* M is the molecular mass
* $I_A$, $I_B$, and $I_C$ are the principal moments of inertia, obtained directly from the nuclear coordinates (no electronic structure calculation required)
* $\sigma$ is a symmetry number derived from the point group of the molecule, which generally lies between 1-60. Usually this is just left at 1 regardless, out of laziness, because the contribution is order $\ln(\sigma)$
* $\omega_i$ are the calculated vibrational frequencies

You of course may also find these expressions in essentially any book covering thermochemistry; I write them down here in condensed form simply to emphasize that **everything here is analytically tractable!** There is really no need to be afraid of using these equations by hand should you need to.

Some important "cannonical" observations on the above:

* You will notice that most of the internal energy terms go to zero at 0K, with 2 exceptions: the electronic energy $E_{elec}$ and the first term of the vibrational energy $\sum_i \frac{1}{2} h \omega_i$. The second of these terms is of course the so-called **"zero point vibrational energy"**, a seemingly endless source of frustration for computational chemists across the globe.
* The rotational and translational degrees of freedom both contribute $\frac{3}{2} k T$, **$\frac{1}{2} k T$ of internal energy per degree of freedom**. It is interesting to note that these terms are directly cancelled out by the trailing entropy contributions!
* The entropy of the translational and rotational degrees of freedom scale with the mass and the moments of inertia of the molecule, respectively. **Heavier particles have more favorable entropy contributions**.

## Back to the Proton: Deriving the Ideal Gas Law

We are now ready to evaluate the molecular chemical potential of our proton in a box:

$$
\mu_{p^+} = U_{trans} + PV - TS_{trans} 
$$

Plugging in our values, we get that:

$$
\mu_{p^+} = \frac{3}{2} k T + PV - T(k \ln \left( \left( \frac{2 \pi M k T}{h^2} \right)^\frac{3}{2} V \right) + k\frac{3}{2})
$$

$$
= PV - k T \ln \left( \left( \frac{2 \pi M k T}{h^2} \right)^\frac{3}{2} V \right)
$$
 
So we see that there is a balancing act here! The pressure component wants to minimize the volume, but the translational entropy of the proton wants to maximize it.

We infer that there must be some finite volume at which this balanced is reached (finite because the entropy term is unbounded as V->0). To do so, we can take a derivative and set it equal to zero. 

To simplify the math, we make the substitution $\lambda = \left( \frac{2 \pi M k T}{h^2} \right)^\frac{-1}{2} $ Using the chain rule for our derivative, we get

$$
\frac{\partial \mu}{\partial V} = P - kT (\frac{1}{\lambda^3})(\frac{1}{\frac{V}{\lambda^3}}) = P - \frac{kT}{V} = 0
$$

Rearranging, you will note that this is simply the one-particle ideal gas law!

$$
P V = k T
$$

Thus, **the ideal gas law is a direct result of the minimization of the chemical potential with respect to the volume for a system with only translational degrees of freedom**.

I think that's really cool! We were able to treat our proton completely quantum-statistically and pull out an observed behavior of gasses in the real world.

Unfortunately this post is getting a bit long, so I am choosing to end here for now (I have a WPE to prepare for!). To all of you who got this far, I hope you found it useful. In future posts I plan to cover:

* Chemical reactions with more than one type particle
* Chemical reactions of moles of particles
* The origin of chemical equlibrium
* Solvent models and how they affect the free energy

Thanks for reading!

