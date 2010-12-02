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

namespace Volkszaehler\Model;

use Doctrine\Common\Collections\ArrayCollection;
use Volkszaehler\Model;

/**
 * Data entity
 *
 * @author Steffen Vogel <info@steffenvogel.de>
 * @package default
 * @todo rename? Bsp: DataSample, Sample, Reading
 *
 * @Entity
 * @Table(
 * 		name="data",
 * 		uniqueConstraints={
 * 			@UniqueConstraint(name="unique_timestamp", columns={"timestamp", "channel_id"})
 * 		}
 * )
 */
class Data {
	/**
	 * @Id
	 * @Column(type="integer", nullable=false)
	 * @GeneratedValue(strategy="AUTO")
	 *
	 * @todo wait until DDC-117 is fixed (PKs on FKs)
	 */
	protected $id;

	/**
	 * Ending timestamp of period in ms since 1970
	 *
	 * @Column(type="bigint")
	 */
	protected $timestamp;

	/**
	 * @Column(type="decimal", precision="5", scale="2")
	 * @todo change to float after DCC-67 has been closed
	 */
	protected $value;

	/**
	 * @ManyToOne(targetEntity="Channel", inversedBy="data")
	 * @JoinColumn(name="channel_id", referencedColumnName="id")
	 */
	protected $channel;

	public function __construct(Model\Channel $channel, $timestamp, $value) {
		$this->channel = $channel;

		$this->value = $value;
		$this->timestamp = $timestamp;
	}

	public function toArray() {
		return array('channel' => $this->channel, 'timestamp' => $this->timestamp, 'value' => $this->value);
	}

	/**
	 * setter & getter
	 */
	public function getValue() { return $this->value; }
	public function getTimestamp() { return $this->timestamp; }
	public function getChannel() { return $this->channel; }
}

?>
