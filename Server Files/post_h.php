<?php
$user_name = $_POST["user_name"];
//$user_name = "1e:4d:70:af:f8:9d*-32*1c:5f:2b:da:78:ec*-64*1e:96:e6:3d:e2:df*-82*";
$output = shell_exec("echo_client.py "  .$user_name);
echo $output;


?>