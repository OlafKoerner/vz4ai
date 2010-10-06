<?php
/**
 * @copyright Copyright (c) 2010, The volkszaehler.org project
 * @package default
 * @license http://www.opensource.org/licenses/gpl-license.php GNU Public License
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

namespace Volkszaehler\View;

use Volkszaehler\Interpreter;

use Volkszaehler\Iterator;

use Volkszaehler\Model;
use Volkszaehler\View\HTTP;
use Volkszaehler\Util;

/**
 * Base class for all view formats
 *
 * @package default
 * @author Steffen Vogel <info@steffenvogel.de>
 *
 */
abstract class View {
	/**
	 * @var integer round all values to x decimals
	 */
	const PRECISSION = 5;

	/**
	 * @var HTTP\Request
	 * @todo do we need this? why public? not via getter?
	 */
	public $request;

	/**
	 * @var HTTP\Response
	 */
	protected $response;

	/**
	 * Constructor
	 *
	 * @param HTTP\Request $request
	 * @param HTTP\Response $response
	 */
	public function __construct(HTTP\Request $request, HTTP\Response $response) {
		$this->request = $request;
		$this->response = $response;

		// error & exception handling by view
		set_exception_handler(array($this, 'exceptionHandler'));
		set_error_handler(array($this, 'errorHandler'));
	}

	public function __desctruct() {
		restore_exception_handler();
		restore_error_handler();
	}

	/**
	 * error & exception handling
	 */
	final public function errorHandler($errno, $errstr, $errfile, $errline) {
		$this->exceptionHandler(new \ErrorException($errstr, 0, $errno, $errfile, $errline));
	}

	final public function exceptionHandler(\Exception $exception) {
		$this->addException($exception, Util\Debug::isActivated());

		$code = ($exception->getCode() == 0 && HTTP\Response::getCodeDescription($exception->getCode())) ? 400 : $exception->getCode();
		$this->response->setCode($code);
		$this->send();

		die();
	}

	public function send() {
		if (Util\Debug::isActivated()) {
			$this->addDebug(Util\Debug::getInstance());
		}

		$this->render();
		$this->response->send();
	}

	/**
	 * Add object to output
	 *
	 * @param mixed $data
	 */
	public function add($data) {
		if (isset($data)) {
			if ($data instanceof Interpreter\InterpreterInterface) {
				$this->addData($data);
			}
			elseif ($data instanceof Model\Entity) {
				$this->addEntity($data);
			}
			elseif ($data instanceof \Exception) {
				$this->addException($data);
			}
			elseif ($data instanceof Util\Debug) {
				$this->addDebug($data);
			}
			else {
				throw new \Exception('Can\'t show ' . get_class($data));
			}
		}
	}

	/**
	 * Sets caching mode for the browser
	 *
	 * @todo implement remaining caching modes
	 * @param $mode
	 * @param integer $value timestamp in seconds or offset in seconds
	 */
	public function setCaching($mode, $value) {
		switch ($mode) {
			case 'modified':	// Last-modified
				//$this->response->setHeader('Last-Modified', gmdate('D, d M Y H:i:s', $value) . ' GMT');

			case 'etag':		// Etag
				throw new Exception('This caching mode is not implemented');

			case 'expires': 	// Expire
				$this->response->setHeader('Expires', gmdate('D, d M Y H:i:s', $value) . ' GMT');
				break;

			case 'age':			// Cache-control: max-age=
				$this->response->setHeader('Cache-control', 'max-age=' . $value);
				break;

			case 'off':
			case FALSE:
				$this->response->setHeader('Cache-control', 'no-cache');
				$this->response->setHeader('Pragma', 'no-cache');

			default:
				throw new Exception('Unknown caching mode');
		}
	}

	protected abstract function render();

	protected abstract function addData(Interpreter\InterpreterInterface $data);
	protected abstract function addEntity(Model\Entity $entity);
	protected abstract function addException(\Exception $exception);
	protected abstract function addDebug(Util\Debug $debug);
}

?>
