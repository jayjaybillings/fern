<?php

$contents = file_get_contents("fernOut.txt");
//print_r($contents);

$cleanStart = explode("---startOutput---\n---outputcount---\n", $contents);
$cleanEnd = explode("---endOutput---\n", $cleanStart[1]);
$outIntervals = explode("---outputcount---\n", $cleanEnd[0]);

//split up Species data from each interval from the universal data (t, dt, T9, rho) of the same interval
foreach($outIntervals as $outIntID => $outIntVal) {
    $splitSpeciesandUniv[$outIntID] = explode("---startUniversaldata---", $outIntVal);
    $universalData[$outIntID] = $splitSpeciesandUniv[$outIntID][1];
    $speciesData[$outIntID] = $splitSpeciesandUniv[$outIntID][0];
    //Generate Master Y, Z, N, Fplus, Fminus arrays for each interval and each species.
    $parseY[$outIntID] = explode("Y: ", $speciesData[$outIntID]);
    unset($parseY[$outIntID][0]);
    foreach($parseY[$outIntID] as $speciesID => $messyYVal) {
        $splodemessyYVal[$speciesID] = explode("\nZ: ", $messyYVal);
        $masterY[$outIntID][$speciesID] = $splodemessyYVal[$speciesID][0];
        $masterY[$outIntID] = array_values($masterY[$outIntID]);
        $parseZ[$outIntID][$speciesID] = $splodemessyYVal[$speciesID][1];
    }
    foreach($parseZ[$outIntID] as $speciesID => $messyZVal) {
        $splodemessyZVal[$speciesID] = explode("\nN: ", $messyZVal);
        $masterZ[$outIntID][$speciesID] = $splodemessyZVal[$speciesID][0];
        $masterZ[$outIntID] = array_values($masterZ[$outIntID]);
        $parseN[$outIntID][$speciesID] = $splodemessyZVal[$speciesID][1];
    }
    foreach($parseN[$outIntID] as $speciesID => $messyNVal) {
        $splodemessyNVal[$speciesID] = explode("\nFplus: ", $messyNVal);
        $masterN[$outIntID][$speciesID] = $splodemessyNVal[$speciesID][0];
        $masterN[$outIntID] = array_values($masterN[$outIntID]);
        $parseFplus[$outIntID][$speciesID] = $splodemessyNVal[$speciesID][1];
    }
    foreach($parseFplus[$outIntID] as $speciesID => $messyFplusVal) {
        $splodemessyFplusVal[$speciesID] = explode("\nFminus: ", $messyFplusVal);
        $masterFplus[$outIntID][$speciesID] = $splodemessyFplusVal[$speciesID][0];
        $masterFplus[$outIntID] = array_values($masterFplus[$outIntID]);
        $parseFminus[$outIntID][$speciesID] = $splodemessyFplusVal[$speciesID][1];
    }
    foreach($parseFminus[$outIntID] as $speciesID => $messyFminusVal) {
        $masterFminus[$outIntID][$speciesID] = str_replace("\n","",$parseFminus[$outIntID][$speciesID]);
        $masterFminus[$outIntID] = array_values($masterFminus[$outIntID]);
    }

    //Generate Master t, dt, T9, and rho arrays for each interval
    $parseTime[$outIntID] = explode("time: ", $universalData[$outIntID]);
    unset($parseTime[$outIntID][0]);
    $splodemessyTimeVal = explode("\ndeltat: ", $parseTime[$outIntID][1]);
    $masterTime[$outIntID] = $splodemessyTimeVal[0];
    $parseDeltat[$outIntID] = $splodemessyTimeVal[1];

    $splodemessyDeltatVal = explode("\nT9: ", $parseDeltat[$outIntID]);
    $masterDeltat[$outIntID] = $splodemessyDeltatVal[0];
    $parseT9[$outIntID] = $splodemessyDeltatVal[1];

    $splodemessyT9Val = explode("\nrho: ", $parseT9[$outIntID]);
    $masterT9[$outIntID] = $splodemessyT9Val[0];
    $parseRho[$outIntID] = $splodemessyT9Val[1];

    $splodemessyRhoVal = explode("\nsumX: ", $parseRho[$outIntID]);
    $masterRho[$outIntID] = $splodemessyRhoVal[0];
    $parseSumX[$outIntID] = $splodemessyRhoVal[1];

    $masterSumX[$outIntID] = str_replace("\n","", $parseSumX[$outIntID]);

} //end foreach($outIntID)


//build final output
$finalOut = "-----------------------------------------------------------------
Particles=1.0E30 maxLight=1000 tripleAlphaOnly=false normX=false
noNeutronsFlag=false Zmax=34 Zmin=2 nT=1.37142873E31 T9=7.00000000
diagnoseII=false tbeg=2.2273E-6 limitByFluxes=false fluxFracThresh=1.0E-16
E/A(init)=0.0000 Mode=Asy SF=0.01 MassTol=1.0E-7 Ymin=0.0
-----------------------------------------------------------------
Everything before the triple dollar sign is ignored, as are blank lines

$$$\n\n43 16 1.3714287281634087E31\n0 0 0 0 0\n0 0 0 0 0\n0 0 0 0 0\n\nPlot_Times:\n";
foreach($masterTime as $v) {
$finalOut .= $v." ";
}
$finalOut .= "\n\nIntegrated_Energy(time):\n";
//will replace with true integrated energy when I have it...
foreach($masterTime as $v) {
$finalOut .= "1.0000 ";
}
$finalOut .= "\n\ndE/dt(time):\n";
foreach($masterTime as $v) {
$finalOut .= "1.0000 ";
}
$finalOut .= "\n\ndt(time):\n";
foreach($masterDeltat as $v) {
$finalOut .= $v." ";
}
$finalOut .= "\n\nT9(time):\n";
foreach($masterT9 as $v) {
$finalOut .= $v." ";
}
$finalOut .= "\n\nrho(time):\n";
foreach($masterRho as $v) {
$finalOut .= $v." ";
}
$finalOut .= "\n\nYe(time):\n";
foreach($masterTime as $v) {
$finalOut .= "0.5000 ";
}
$finalOut .= "\n\nsumX(time):\n";
foreach($masterSumX as $v) {
$finalOut .= $v." ";
}

$finalOut .= "\n\nAbundances_Y(Z,N,time):\n\n";

foreach($masterY[0] as $speciesID => $Yval) {
$finalOut .= $masterZ[0][$speciesID]." ";
$finalOut .= $masterN[0][$speciesID]."\n";
    foreach($masterY as $outIntID => $species) {
        $finalOut .= $species[$speciesID]." ";
    }
$finalOut .= "\n\n";
}

$finalOut .="JavaAsy (Optional_comment) 501 0.01 1.0E-7\n0.0 3 3 2 1.0067596421182161\n1.3714287281634087E31 -1.0344686097237472E31 false true 7.0\n1.0E8 jin/torch47Profile.inp 3 jin/alpha3.inp 1\njin/nova125DRatesReduced.inp 2 jin/abundance.inp false\n\n";

$finalOut .= "Fplus(Z,N,time):\n\n";

foreach($masterFplus[0] as $speciesID => $Fplusval) {
$finalOut .= $masterZ[0][$speciesID]." ";
$finalOut .= $masterN[0][$speciesID]."\n";
    foreach($masterFplus as $outIntID => $species) {
        $finalOut .= $species[$speciesID]." ";
    }
$finalOut .= "\n\n";
}

$finalOut .= "Fminus(Z,N,time):\n\n";

foreach($masterFminus[0] as $speciesID => $Fminusval) {
$finalOut .= $masterZ[0][$speciesID]." ";
$finalOut .= $masterN[0][$speciesID]."\n";
    foreach($masterFminus as $outIntID => $species) {
        $finalOut .= $species[$speciesID]." ";
    }
$finalOut .= "\n\n";
}

print_r($finalOut);

?>
