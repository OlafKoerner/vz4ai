<?php
/**
 * middleware bootstrap entrypoint
 *
 * @author Steffen Vogel <info@steffenvogel.de>
 * @copyright Copyright (c) 2011-2020, The volkszaehler.org project
 * @license https://www.gnu.org/licenses/gpl-3.0.txt GNU General Public License version 3
 */
/*
 * This file is part of volkzaehler.org
 *
 * volkzaehler.org is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * volkzaehler.org is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with volkszaehler.org. If not, see <http://www.gnu.org/licenses/>.
 */

namespace Volkszaehler;


//OKO 2023-10-03 to allow cross domain access to reach botte REST api
//https://stackoverflow.com/questions/20035101/why-does-my-javascript-code-receive-a-no-access-control-allow-origin-header-i
//<?php header('Access-Control-Allow-Origin: *'); ?>


use Symfony\Component\HttpFoundation\Request;

define('VZ_DIR', realpath(__DIR__ . '/..'));

// default response if things go wrong
http_response_code(500);

require VZ_DIR . '/lib/bootstrap.php';

$router = new Router();

// create Request from PHP global variables
$request = Request::createFromGlobals();

// handle request
$response = $router->handle($request);
$response->send();
