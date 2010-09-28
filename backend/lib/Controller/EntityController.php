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

use Volkszaehler\Definition;

use Volkszaehler\Util;
use Volkszaehler\Model;

/**
 * Entity controller
 *
 * @author Steffen Vogel <info@steffenvogel.de>
 * @package default
 */
class EntityController extends Controller {
	/**
	 * Get entity
	 *
	 * @param string $identifier
	 */
	public function get($uuid) {
		if (!Util\UUID::validate($uuid)) {
			throw new \Exception('Invalid UUID: ' . $uuid);
		}

		$dql = 'SELECT a, p
				FROM Volkszaehler\Model\Entity a
				LEFT JOIN a.properties p
				WHERE a.uuid = ?1';

		$q = $this->em->createQuery($dql);
		$q->setParameter(1, $uuid);

		try {
			return $q->getSingleResult();
		} catch (\Doctrine\ORM\NoResultException $e) {
			throw new \Exception('No entity found with UUID: ' . $uuid);
		}
	}

	/**
	 * Delete entity by uuid
	 */
	public function delete($identifier) {
		$entity = $this->get($identifier);

		$this->em->remove($entity);
		$this->em->flush();
	}

	/**
	 * Edit entity properties
	 */
	public function edit($identifier) {
		$entity = $this->get($identifier);
		$this->setProperties($entity);
		$this->em->flush();

		return $entity;
	}

	protected function setCookie(Model\Entity $entity) {
		if ($uuids = $this->view->request->getParameter('uuids', 'cookies')) {
			$uuids = Util\JSON::decode($uuids);
		}
		else {
			$uuids = new Util\JSON();
		}

		// add new UUID
		$uuids->append($entity->getUuid());

		// remove duplicates
		$uuids->exchangeArray(array_unique($uuids->getArrayCopy()));

		// send new cookie to browser
		setcookie('uuids', $uuids->encode());
	}

	protected function unsetCookie(Model\Entity $entity) {
		if ($uuids = $this->view->request->getParameter('uuids', 'cookies')) {
			$uuids = Util\JSON::decode($uuids);
		}
		else {
			$uuids = new Util\JSON();
		}

		// remove old UUID
		$uuids->exchangeArray(array_filter($uuids->getArrayCopy, function($uuid) use ($entity) {
			return $uuid != $entity->getUuid();
		}));

		// send new cookie to browser
		setcookie('uuids', $uuids->encode());
	}

	protected function setProperties(Model\Entity $entity) {
		foreach ($this->view->request->getParameters() as $parameter => $value) {
			if (Definition\PropertyDefinition::exists($parameter)) {
				if ($value == '') {
					$entity->unsetProperty($parameter, $this->em);
				}
				else {
					$entity->setProperty($parameter, $value);
				}
			}
		}
	}
}

?>