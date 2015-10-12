cccccccccccccccccccccccccccccccccccccccccccccccccccc
cccccccccccccccccccccccccccccccccccccccccccccccccccc
 
   This is a short description about the
   content of the two files:
   1) reaclib.nosmo
   2) winvn.nosmo
------------------------------------------------------
                reaclib.nosmo
 
   This is a limited update of a previous version. The previous version
   was the 1991 version of reaclib, updated to z=46 in 1995 by
   Ch. Freiburghaus and T. Rauscher. It can be found online at
   http://ie.lbl.gov/astro/friedel.html . In this update, the
   theoretical NON-SMOKER rates were inserted and replaced previous
   theoretical values. The NON-SMOKER rates include targets up to z=83.
   As mass formula, the FRDM was chosen.

   reaclib is (attempted to be) a complete library of nuclear reaction
   rates from z=0 to z=46(83). It consists of 8 different sections of
   different reactions of the types:
   1: a -> b  essentially beta-decays and electron
              captures
   2: a -> b + c   mainly photodisintegrations or
                   beta delayed neutron emission
   3: a -> b + c + d  like inverse triple-alpha
                   or beta-delayed two neutron
                   emsission
------------------------------------------------------
   4: a + b -> c   capture reactions
   5: a + b -> c + d   particle exchange like (p,n)
   6: a + b -> c + d + e
   7: a + b -> c + d + e + f
------------------------------------------------------
   8: a + b + c -> d (+ e ) three particle reactions
                            like triple-alpha

   each rate is described by three lines. the first line indicates

   1: the participating nuclei

   2: the source of the reaction
    a4 like cf88, wies, baka, fkth etc.

   cf88 caughlan and fowler 1988
   wies several papers involving michael wiescher
        joachim goerres, karl-heinz langanke and
        myself
   laur laura van wormer, wiescher, goerres, iliades, thielemann 
        1994
   bb92 rauscher, applegate, cowan, thielemann, wiescher 1994
   baka bao and kaeppeler 1987
   rolf claus rolfs and collaborators
   wawo wallace and woosley 1980
   mafo malaney and fowler 1988
    wfh wagoner, fowler, hoyle 1964
    wag wagoner 1969 (these last two sources only if
        absolutely nothing else was available

    --- all these rates are based on experiments ----
 
   fkth theoretical rates calculated with statistical
        model program SMOKER ( thielemann, arnould, truran 1987,
                       see also cowan, thielemann, truran 1991 (phys. rep.)
   rath theoretical rates calculated with statistical
        model program NON-SMOKER ( rauscher and thielemann, ADNDT 76 (2000) 1.)
    ----------------------------------------------------
   bet- experimental beta-minus rates
   bet+      "        "    plus   "
    bec experimental electron-capture rates
        this is a preliminary stage and should be
        generalized to stellar el-capture later
     ec electron capture rates !!!!!!!!!!!!!!!!!!!!!!!!
        !!!! these (and only ec) should be multiplied with
        rho*Ye!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ----------------------------------------------------
   btyk theoretical beta-plus rates by takahashi, yamada, kondo
        1973, 1980
   bkmo theoretical beta-minus rates by klapdor, metzinger, oda
        1984
   mo92 theoretical beta-minus rates by moeller 1992

   3: the type of recation for future treatment of screening
      r: resonant n: nonresonant
      all experimental rates habe been split into resonant and
      nonresonant parts, thus the total reaction rate is the
      sum of n + r !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      because the fitting range was extended, t=1.e7K to 1.e10K,
      some experimental resonant rates were again split into
      several terms to achieve a the necessary accuracy.
      then the total rate is n + r1 + r2 !!!!!!!!!!!!!!!!!!!!!
      !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      (the only exception where a split into resonant and non
      resonant terms has not always been taken seriously was for
      neutron captures, where electron screening anyway does
      not play any role)

   4: if the rate is calculated from the inverse reaction this
      is indicated by a "v". this means that in a network 
      calculation this rate has to be multiplied by the ratio
      of the (normalized) partition functions of the participating
      nuclei
      e.g. the reaction b -> c + d or a + b -> c + d
      would have the revised rate  = rate (fit)  * part(d)/part(b).
      here it is assumed that a and c are neutrons, protons or
      alphas with part=1 (normalized to the ground state). 
      the partition functions of all nuclei (normalized to the
      ground state, i.e. /(2j0+1))  are found in winvn (see that 
      section). the ratio of the ground state statistical weights
      is already included in the fitted rates.
 
   5: the Q-value of the reaction in MeV

                  ---------------------
      the second and third line for each rate give seven coefficients

      the rate is calculated by
 
      rate=exp(a1 + a2/t9 + a3/t913 + a4*t913 + a5*t9 + a6*t953 + a7*ln(t9))
      and corresponds to ln(2)/t12 for decays, NA<ab> for two-body
      reactions and NA**2<abc> for three-body reactions.

   =================================================================
  ===================================================================
                        winvn

      the dataset winvn gives a list of all nuclei which are incorporated
      into reaclib. they are essentially nuclei from n and p up to
      Bi and for each element all isotopes between neutron and proton
      drip line (FRDM mass formula). note that al26 appears as
      al26, al-6, and al*6. In a network you should make the choice
      al26 or al-6 and al*6, dependent if you want the nucleus as such
      or the ground and metastable state seperately. The choice for
      al-6 and al*6 should only be made for t9<1 when these states are
      not yet in thermal equlibrium.
      following the list of all nuclei you find again each nucleus with
      all its specifications
      line 1: name , A, Z, N, ground state spin, nuclear mass excess
      line 2 and 3: partition function at 24 values of t9. you can find
      these values in the second line of winvn given in units of t7
      i.e. t9=0.1, 0.15, 0.2     ...... 10. (note that value 24 is given
      in units of t8). below t9=0.1 one can safely assume that part=1.
      for the usage with inverse reactions (v) in reaclib these values
      should be logarithmically interpolated for the apropriate
      temperatures.

   <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

      let me finally say something about the validity and accuracy
      of the rates.
      all experimental rates (everything except fkth) are
      fitted in the range t=1.e7K to 1.e10K. 
      that means these parametrized rates can be used from hydrogen 
      burning onwards. all fkth rates are fitted between t=1.e8K and
      1.e10K, they behave asymptotically correct down to 1.e7
      in the sense that charged particle rates will decrease further
      and neutron captures stay roughly constant. because charged
      particle rates for heavy nuclei in the fkth-part are anyway
      negligible below 1.e8K, the whole set an be used safely in the
      range 1.e7 to 1.e10. but do not use these rates below 1.e7K,
      funny things might happen. if you really need the rates at these
      low temperatures (which is essentially only deuterium burning
      in pre-main sequence stars) use the original expression from
      caughlan and fowler etc.

      the fits compare with the experimental expressions (e.g.
      cf88) in a vast amount of cases better than 1%. With
      the exception of a few cases the agreement is better
      than 10% and only some rates have uncertainties up to
      30%. on the other side the cf88 rates are not completely
      exact either, many higher energy resonances are lumped
      into one analytical term and an accuracy better than
      30% is usually achieved for those cases ( Harris, private
      communication). thus these uncertainties are comparable.


                                  Have Fun and Good Luck
 
                              Thomas Rauscher, Friedel Thielemann

