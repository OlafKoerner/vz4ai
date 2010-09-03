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

namespace Volkszaehler\Controller;

use Volkszaehler\Util;
use Volkszaehler\Model;

/**
 * Channel controller
 *
 * @author Steffen Vogel <info@steffenvogel.de>
 * @package default
 */
class ChannelController extends EntityController {
	/**
	 * Add channel
	 */
	public function add() {
		$channel = new Model\Channel($this->view->request->getParameter('type'));

		foreach ($this->view->request->getParameters() as $parameter => $value) {
			if (Model\PropertyDefinition::exists($parameter)) {
				$property = new Model\Property($channel, $parameter, $value);
				$channel->setProperty($property);
			}
		}

		$this->em->persist($channel);
		$this->em->flush();

		$this->view->addChannel($channel);
	}
}

?>