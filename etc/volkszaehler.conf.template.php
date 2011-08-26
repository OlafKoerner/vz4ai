<?php
/**
 * Configuration template
 *
 * You should use this file to obtain your custom configuration
 * new parameters should be documented
 *
 * @copyright Copyright (c) 2011, The volkszaehler.org project
 * @author Steffen Vogel <info@steffenvogel.de>
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

/**
 * PDO driver for Doctrine DBAL
 *
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en#getting-a-connection:driver
 */
$config['db']['driver']				= 'pdo_mysql';

/**
 * Hostname of database server
 * 
 * Use 'localhost' for the machine your webserver is running on.
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en
 */
$config['db']['host']				= 'localhost';

/**
 * Username for the database server
 *
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en
 */
$config['db']['user']				= 'vz';

/**
 * Password for the database server
 *
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en
 */
$config['db']['password']			= 'demo';

/**
 * @var string database name
 *
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en
 */
$config['db']['dbname']				= 'volkszaehler';

/**
 * Administration credentials
 *
 * For administration tasks (used by doctrine cli and the setup script)
 * the following $['db']['admin'] settings will be merged with $config['db']
 */
//$config['db']['admin']['user']		= 'vz_admin';
//$config['db']['admin']['password']		= 'admin_demo';

/**
 * Path of the SQLite database
 * 
 * See Doctrine doc for details: http://www.doctrine-project.org/projects/dbal/2.0/docs/reference/configuration/en
 */
//$config['db']['path']				= 'volkszaehler';

/**
 * Vendor libs
 *
 * Set to NULL to use PHP's include path
 * See PHP doc for details: http://www.php.net/manual/en/ini.core.php#ini.include-path
 */
//$config['lib']['doctrine']			= VZ_DIR . '/lib/vendor/Doctrine';
//$config['lib']['jpgraph']			= VZ_DIR . '/lib/vendor/JpGraph';

/**
 * Timezone for the middleware
 *
 * See PHP doc for details: http://www.php.net/manual/de/timezones.php
 * http://www.php.net/manual/de/datetime.configuration.php#ini.date.timezone
 */
//$config['timezone']				= 'Europe/Berlin';

/**
 * Locale used for regular expressions
 *
 * See PHP doc for details: http://php.net/manual/de/function.setlocale.php
 */
$config['locale']				= array('de_DE', 'en_US', 'C');

/**
 * Array of colors for plot series
 *
 * @attention Only used by jpGraph for server-side plotting!
 */
$config['colors'] = array('#83CAFF', '#7E0021', '#579D1C', '#FFD320', '#FF420E', '#004586', '#0084D1', '#C5000B', '#FF950E', '#4B1F6F', '#AECF00', '#314004');

/**
 * Developer mode
 *
 * This disables all caching mechanisms and enabled debugging by default
 */
$config['devmode']				= FALSE;

/**
 * Debugging level
 *
 * Set to > 0 to enable debug messages by default
 */
$config['debug']				= 0;

?>
