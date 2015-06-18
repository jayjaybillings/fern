<?php

$reacLib = file('rateLibrary_atmosCHASER.data');
foreach($reacLib as $shine) {
  $i++;
  if(($i-1) % 8 == 0) {
    //this is line 1 of each group of 8
    $reacNum++;
    $entry = explode(" ", $shine);
    if($reacNum < 27) {
      printf($entry[0]." 1 ".$entry[1]." ".$entry[2]." ".$entry[3]." ".$entry[4]." ".$entry[5]." ".$entry[6]." ".$entry[7]." ".$entry[8]." ".$entry[9]." ".$entry[10]);
    } else if($reacNum >=27){
      printf($entry[0]." 0 ".$entry[1]." ".$entry[2]." ".$entry[3]." ".$entry[4]." ".$entry[5]." ".$entry[6]." ".$entry[7]." ".$entry[8]." ".$entry[9]." ".$entry[10]);
    }
    //printf("line ". $i . " is not a multiple of 8\n");
  } else if (($i-3) % 8 == 0) {
    //this is line 3 of each group of 8
    $entry = array_filter(array_map('trim',explode(" ", $shine)));
//    print_r($entry);

    foreach($entry as $print) {
      printf("1 ");
    }
    printf("\n");
  } else if (($i-4) % 8 == 0) {
    //this is line 4 of each group of 8
  } else if (($i-5) % 8 == 0) {
    //this is line 5 of each group of 8
    $entry = array_filter(array_map('trim',explode(" ", $shine)));
    foreach($entry as $print) {
      printf("1 ");
    }
    printf("\n");
  } else if (($i-6) % 8 == 0) {
    //this is line 6 of each group of 8
  } else {
    printf($shine);
  }
}

?>
