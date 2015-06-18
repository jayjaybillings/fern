<?php

$filename = 'reactions.txt';
$reacLib = file('rateLibrary_atmos.data');
$contents = file($filename);

foreach($contents as $line) {
    $reactionID = explode("k", explode("=", $line)[0])[1]; 
    $eqn = str_replace("\n", "", explode("=", $line)[1]); 
      $A = 0;
      $B = 0;
      $C = 0;
      $D = 0;
      $E = 0;
      $F = 0;
      $G = 0;
      $a = 0;
      $b = 0;
      $c = 0;
      $d = 0;
      $e = 0;
      $t = 1;
      $u = 0;
      $v = 1;
      $w = 0;
      $x = 1;
      $Q = 1;
      $R = 0;
    if(strpos($eqn, 'exp(') === false) {
      $A = $eqn;
      $finalTerm = 0;
    } else {
      $C = explode("exp(", $eqn)[0];
      $finalTerm = explode("exp(",$eqn)[1];
    }

    if($finalTerm != 0) {
      $b = explode("/", $finalTerm)[0]; 
    }
    $i = 0;
    $lineSet;
    $paramLine = (8*$reactionID)+202;
    foreach($reacLib as $shine) {
      $i++;
      if($i == $paramLine && empty($lineSet[$paramLine])) {
        $lineSet[$i] = 1;
        //echo "$reactionID: A=".$A.",C=".$C.", a=".$a."\n";
        echo $A." ".$B." ".$C." ".$D." ".$E." ".$F." ".$G." ".$a." ".$b." ".$c." ".$d." ".$e." ".$t." ".$u." ".$v." ".$w." ".$x." ".$Q." ".$R."\n";
      } else if ($i < $paramLine && empty($lineSet[$i])){
        echo $shine;
        $lineSet[$i] = 1;
      }

    }
    
}

?>
